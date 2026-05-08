////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94

#include <legion.h>

#include <list>
#include <set>
#include <map>
#include <vector>
#include <functional>
#include <mutex>
#include <cassert>
#include <memory>
#include <atomic>
#include <fstream>

// ── ASTM Config (Legion) ────────────────────────────────────────────────────

#define ASTM_MUTEX std::mutex
#define ASTM_LOCK  std::unique_lock<std::mutex>

#define ASTM_TEST  assert
#define ASTM_REPORT 0

namespace astm
{

// ── Legion runtime context (thread-local so child tasks get their own) ──────

struct legion_context
{
    static inline thread_local Legion::Runtime* runtime = nullptr;
    static inline thread_local Legion::Context  ctx     = {};

    static void set(Legion::Runtime* rt, Legion::Context c)
    {
        runtime = rt;
        ctx     = c;
    }
};

// ── Function registry : stores std::function<void()> keyed by uint64_t ──────
//    Used to shuttle arbitrary callables into registered Legion tasks.

struct func_registry
{
    static inline std::mutex                                    mtx_;
    static inline std::map<uint64_t, std::function<void()>>    funcs_;
    static inline std::atomic<uint64_t>                         next_id_{0};

    static uint64_t register_func(std::function<void()> f)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        uint64_t id = next_id_++;
        funcs_[id]  = std::move(f);
        return id;
    }

    static void execute(uint64_t id)
    {
        std::function<void()> f;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            auto it = funcs_.find(id);
            assert(it != funcs_.end());
            f = std::move(it->second);
            funcs_.erase(it);
        }
        f();
    }
};

// ── Task IDs ────────────────────────────────────────────────────────────────

enum
{
    ASTM_FUNC_TASK_ID = 9900   // keep well away from user task-IDs
};

// ── Generic "run a registered function" task implementation ─────────────────

inline void astm_func_task_impl(
    const Legion::Task*                         task,
    const std::vector<Legion::PhysicalRegion>&  /*regions*/,
    Legion::Context                             ctx,
    Legion::Runtime*                            runtime)
{
    legion_context::set(runtime, ctx);
    assert(task->arglen == sizeof(uint64_t));
    uint64_t id = *reinterpret_cast<const uint64_t*>(task->args);
    func_registry::execute(id);
}

// ── Static pre-registration (runs once before Runtime::start) ───────────────

namespace detail
{
    struct task_registrar
    {
        task_registrar()
        {
            Legion::TaskVariantRegistrar registrar(ASTM_FUNC_TASK_ID,
                                                   "astm_func_task");
            registrar.add_constraint(
                Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
            Legion::Runtime::preregister_task_variant<astm_func_task_impl>(
                registrar, "astm_func_task");
        }
    };
    static inline task_registrar __astm_task_reg{};
} // namespace detail

// ── legion_future<void>  ────────────────────────────────────────────────────
//    Thin wrapper around Legion::Future that provides .get() and .then(F).

template <typename T>
struct legion_future;        // only void specialization used by ASTM

template <>
struct legion_future<void>
{
    Legion::Future fut_;
    bool           has_fut_;

    // "ready" future – nothing to wait on
    legion_future() : fut_(), has_fut_(false) {}

    // wraps an actual Legion::Future
    explicit legion_future(Legion::Future f) : fut_(f), has_fut_(true) {}

    legion_future(const legion_future&)            = default;
    legion_future(legion_future&&)                 = default;
    legion_future& operator=(const legion_future&) = default;
    legion_future& operator=(legion_future&&)      = default;

    void get()
    {
        if (has_fut_)
            fut_.get_void_result();
    }

    /// Chain: when *this completes, execute f (which takes no / ignored args).
    /// Returns a new future that completes after f returns.
    template <typename F>
    legion_future<void> then(F f)
    {
        // wrap the callable into the registry
        std::function<void()> callable(std::move(f));
        uint64_t id = func_registry::register_func(std::move(callable));

        Legion::TaskLauncher launcher(
            ASTM_FUNC_TASK_ID,
            Legion::TaskArgument(&id, sizeof(id)));

        // add the predecessor future so the runtime enforces ordering
        if (has_fut_)
            launcher.add_future(fut_);

        Legion::Future new_fut =
            legion_context::runtime->execute_task(legion_context::ctx,
                                                  launcher);
        return legion_future<void>(new_fut);
    }
};

// ── Helper: create a ready future ───────────────────────────────────────────

inline legion_future<void> make_ready_legion_future()
{
    return legion_future<void>();
}

// ── legion_async : launch a callable as a Legion task, return future ────────

template <typename F, typename... Args>
legion_future<void> legion_async(F&& f, Args&&... args)
{
    auto bound = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    std::function<void()> callable(
        [b = std::move(bound)]() mutable { b(); });

    uint64_t id = func_registry::register_func(std::move(callable));

    Legion::TaskLauncher launcher(
        ASTM_FUNC_TASK_ID,
        Legion::TaskArgument(&id, sizeof(id)));

    Legion::Future fut =
        legion_context::runtime->execute_task(legion_context::ctx, launcher);

    return legion_future<void>(fut);
}

// ── Macros expected by the rest of the ASTM code ────────────────────────────

#define ASTM_FUTURE             astm::legion_future
#define ASTM_FUNCTION           std::function
#define ASTM_ASYNC              astm::legion_async
#define ASTM_MAKE_READY_FUTURE  astm::make_ready_legion_future()

// ═══════════════════════════════════════════════════════════════════════════
//  Core ASTM types (shared_var, transaction, local_var, transaction_future)
// ═══════════════════════════════════════════════════════════════════════════

struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;

    virtual void write(shared_var_base const&) = 0;

    virtual ASTM_LOCK lock() const = 0;

    virtual bool operator==(shared_var_base const&) const = 0;
};

struct transaction;

struct transaction_future
{
    typedef ASTM_FUTURE<void> future_type;

  private:
    transaction* trans_;
    future_type  fut_;

  public:
    transaction_future(transaction* trans)
      : trans_(trans)
      , fut_(ASTM_MAKE_READY_FUTURE)
    {}

    transaction_future(transaction& trans)
      : trans_(&trans)
      , fut_(ASTM_MAKE_READY_FUTURE)
    {}

    template <typename F>
    void then(F f);

    void get()
    {
        fut_.get();
    }
};

// ── shared_var<T> ───────────────────────────────────────────────────────────

template <typename T>
struct shared_var : shared_var_base
{
    typedef ASTM_FUTURE<void> future_type;

    struct local_var
    {
      private:
        transaction*    trans_;
        shared_var_base* var_;

      public:
        local_var(transaction* trans, shared_var_base* var)
          : trans_(trans)
          , var_(var)
        {}

        T get() const;

        operator T const& () const;

        local_var& operator=(shared_var_base const& rhs);

        local_var& operator=(T const& rhs);

        template <typename F>
        void then(F f);
    };

  private:
    T                data_;
    mutable ASTM_MUTEX mtx_;

  public:
    future_type queue;

    shared_var()
      : data_(), mtx_(), queue(ASTM_MAKE_READY_FUTURE) {}

    shared_var(T const& t)
      : data_(t), mtx_(), queue(ASTM_MAKE_READY_FUTURE) {}

    shared_var(T&& t)
      : data_(std::move(t)), mtx_(), queue(ASTM_MAKE_READY_FUTURE) {}

    shared_var(shared_var const& rhs)
      : data_(rhs.data_), mtx_(), queue(ASTM_MAKE_READY_FUTURE) {}

    ~shared_var() {}

    // Locks.
    shared_var_base* clone() const
    {
        auto l = lock();
        return new shared_var(data_);
    }

    // Doesn't lock.
    T const& read() const
    {
        return data_;
    }

    // Doesn't lock.
    void write(T const& rhs)
    {
        data_ = rhs;
    }

    // Doesn't lock.
    void write(shared_var_base const& rhs)
    {
        data_ = dynamic_cast<shared_var const*>(&rhs)->read();
    }

    ASTM_LOCK lock() const
    {
        return ASTM_LOCK(mtx_);
    }

    bool operator==(shared_var_base const& rhs) const
    {
        return data_ == dynamic_cast<shared_var const*>(&rhs)->read();
    }

    local_var get_local(transaction& trans)
    {
        return local_var(&trans, this);
    }
};

// ── transaction ─────────────────────────────────────────────────────────────

struct transaction
{
    std::list<
        std::pair<
            shared_var_base*                        // The shared variable we're reading from
          , std::shared_ptr<shared_var_base>         // The value we read from the variable
        >
    > read_list;

    std::set<
        shared_var_base*                            // The shared variable we're writing to
    > write_set;

    std::list<
        std::pair<
            ASTM_FUTURE<void>*                      // The future we're writing to (NULL → fire-and-forget)
          , ASTM_FUNCTION<void(transaction*)>        // The async action to execute
        >
    > async_list;

    std::map<
        shared_var_base*                            // The shared variable we're reading from
      , std::shared_ptr<shared_var_base>             // Current value of the variable
    > variables;

    void clear()
    {
        read_list.clear();
        write_set.clear();
        async_list.clear();
        variables.clear();
    }

    bool commit_transaction()
    {
        // Algorithm:
        //
        // 1.) Obtain exclusive access to all the variables.
        // 2.) Compare recorded reads against current values (fail if needed).
        // 3.) Perform writes, reading from our internal map.
        // 4.) Perform async operations.
        // 5.) Release exclusive access.

        // 1.) Obtain exclusive access to all the variables.
        std::list<ASTM_LOCK> locks;

        for ( std::pair<shared_var_base*, std::shared_ptr<shared_var_base> > const& var
            : variables)
        {
            assert(var.first != NULL);
            locks.push_back((*var.first).lock());
        }

        // 2.) Compare recorded reads against current values.
        for ( std::pair<shared_var_base*, std::shared_ptr<shared_var_base> > const& var
            : read_list)
        {
            assert(var.first != NULL);
            if (!((*var.first) == (*var.second)))
            {
                clear();
                return false;
            }
        }

        // 3.) Perform writes, reading from our internal map.
        for (shared_var_base* var : write_set)
        {
            assert(var != NULL);
            auto it = variables.find(var);
            assert(it != variables.end());
            (*var).write((*(*it).second));
        }

        // 4.) Perform async operations.
        for ( std::pair<ASTM_FUTURE<void>*, ASTM_FUNCTION<void(transaction*)> >& op
            : async_list)
        {
            if (op.first == NULL)
                ASTM_ASYNC(op.second, this);
            else
                (*op.first) = (*op.first).then(std::bind(op.second, this));
        }

        // 5.) Release exclusive access (RAII via locks list destruction).
        return true;
    }

    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != NULL);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base> > entry(var, nullptr);

        auto result = variables.insert(entry);

        if (result.second == true)
        {
            (*result.first).second.reset((*var).clone());
            read_list.push_back(*result.first);
            return (*(*result.first).second);
        }
        else
            return (*(*result.first).second);
    }

    void write(shared_var_base* var, shared_var_base const& value)
    {
        assert(var != NULL);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base> >
            entry(var, std::shared_ptr<shared_var_base>(value.clone()));

        auto result = variables.insert(entry);

        if (result.second == true)
        {
            write_set.insert(var);
        }
        else
        {
            (*result.first).second = entry.second;
            write_set.insert(var);
        }
    }

    void then(ASTM_FUTURE<void>* fut, ASTM_FUNCTION<void(transaction*)> F)
    {
        std::pair<ASTM_FUTURE<void>*, ASTM_FUNCTION<void(transaction*)> > entry(fut, F);
        async_list.push_back(entry);
    }
};

// ── Out-of-line member definitions for shared_var<T>::local_var ─────────────

template <typename T>
shared_var<T>::local_var::operator T const& () const
{
    return dynamic_cast<shared_var const*>(&trans_->read(var_))->read();
}

template <typename T>
T shared_var<T>::local_var::get() const
{
    return dynamic_cast<shared_var const*>(&trans_->read(var_))->read();
}

template <typename T>
typename shared_var<T>::local_var&
shared_var<T>::local_var::operator=(shared_var_base const& rhs)
{
    trans_->write(var_, rhs);
    return *this;
}

template <typename T>
typename shared_var<T>::local_var&
shared_var<T>::local_var::operator=(T const& rhs)
{
    shared_var tmp(rhs);
    trans_->write(var_, tmp);
    return *this;
}

template <typename T>
template <typename F>
void shared_var<T>::local_var::then(F f)
{
    assert(trans_);
    trans_->then(&dynamic_cast<shared_var*>(var_)->queue, f);
}

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_);
    trans_->then(&fut_, f);
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
