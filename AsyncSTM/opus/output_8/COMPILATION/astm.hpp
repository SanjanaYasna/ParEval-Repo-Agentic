////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Translated to the Legion execution model.
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94

#include "legion.h"

#include <list>
#include <set>
#include <map>
#include <vector>
#include <functional>
#include <mutex>
#include <memory>
#include <cassert>
#include <atomic>
#include <cstring>
#include <fstream>

// ---------------------------------------------------------------------------
// Macro definitions for the Legion backend
// ---------------------------------------------------------------------------
#define ASTM_MUTEX      std::mutex
#define ASTM_LOCK       std::unique_lock<std::mutex>
#define ASTM_FUNCTION   std::function
#define ASTM_TEST       assert
#define ASTM_REPORT     0

namespace astm
{

// ===================================================================
//  Legion infrastructure: context, callback registry, callback task
// ===================================================================

enum ASTMTaskIDs
{
    ASTM_CALLBACK_TASK_ID = 50000 // intentionally high to avoid collisions
};

// Per-task (thread-local) Legion runtime / context.
// Must be set at the entry of every Legion task that uses ASTM.
inline thread_local Legion::Runtime* g_runtime  = nullptr;
inline thread_local Legion::Context  g_context;

inline void init_context(Legion::Runtime* rt, Legion::Context ctx)
{
    g_runtime = rt;
    g_context = ctx;
}

// -------------------------------------------------------------------
// Callback registry – allows us to pass arbitrary std::function<void()>
// objects through Legion TaskArguments (which must be POD).
// -------------------------------------------------------------------
class callback_registry
{
    static inline std::mutex                                mtx_;
    static inline std::map<int64_t, std::function<void()>>  cbs_;
    static inline std::atomic<int64_t>                      next_id_{0};

public:
    static int64_t register_fn(std::function<void()> f)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        int64_t id = next_id_++;
        cbs_[id] = std::move(f);
        return id;
    }

    static std::function<void()> pop_fn(int64_t id)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = cbs_.find(id);
        assert(it != cbs_.end());
        auto f = std::move(it->second);
        cbs_.erase(it);
        return f;
    }
};

// -------------------------------------------------------------------
// The single Legion task that executes registered callbacks.
// Precondition futures (added via TaskLauncher::add_future) are
// explicitly waited on before the callback runs so that .then()
// chaining is honoured.
// -------------------------------------------------------------------
namespace detail
{
    inline void callback_task_body(
        const Legion::Task*                      task,
        const std::vector<Legion::PhysicalRegion>& /*regions*/,
        Legion::Context                          ctx,
        Legion::Runtime*                         rt)
    {
        init_context(rt, ctx);

        // Wait for every precondition future that was attached.
        for (unsigned i = 0; i < task->futures.size(); ++i)
            task->futures[i].get_void_result();

        assert(task->arglen == sizeof(int64_t));
        int64_t cb_id;
        std::memcpy(&cb_id, task->args, sizeof(int64_t));

        auto f = callback_registry::pop_fn(cb_id);
        f();
    }
} // namespace detail

// -------------------------------------------------------------------
// Pre-register the callback task.  Safe to call more than once;
// only the first invocation does real work.
// -------------------------------------------------------------------
inline void register_tasks()
{
    static bool done = false;
    if (done) return;
    done = true;

    Legion::TaskVariantRegistrar reg(ASTM_CALLBACK_TASK_ID, "astm_callback");
    reg.add_constraint(
        Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    reg.set_leaf(true);

    Legion::Runtime::preregister_task_variant<detail::callback_task_body>(
        reg, "astm_callback");
}

// Auto-register when the header is included (runs before main).
namespace detail
{
    struct auto_register_t { auto_register_t() { register_tasks(); } };
}
static astm::detail::auto_register_t __astm_auto_register__;

// ===================================================================
//  legion_future<T>  –  thin wrapper around Legion::Future that
//  provides the .then() / .get() interface expected by the STM code.
// ===================================================================

template <typename T = void>
class legion_future
{
    Legion::Future fut_;
    bool           valid_;      // false  ⇒ logically "ready"

public:
    legion_future()
      : fut_(), valid_(false)
    {}

    explicit legion_future(Legion::Future f)
      : fut_(f), valid_(true)
    {}

    // Copy / move – use compiler defaults (Legion::Future is a handle).
    legion_future(const legion_future&)            = default;
    legion_future(legion_future&&)                 = default;
    legion_future& operator=(const legion_future&) = default;
    legion_future& operator=(legion_future&&)      = default;

    /// Chain a continuation.  Returns a new future that becomes ready
    /// when `f()` completes (and only after *this* future is ready).
    template <typename F>
    legion_future<void> then(F f)
    {
        assert(g_runtime != nullptr);

        int64_t cb_id = callback_registry::register_fn(
            [f]() mutable { f(); });

        Legion::TaskLauncher launcher(
            ASTM_CALLBACK_TASK_ID,
            Legion::TaskArgument(&cb_id, sizeof(int64_t)));

        // The callback task explicitly waits on every attached future.
        if (valid_)
            launcher.add_future(fut_);

        Legion::Future result =
            g_runtime->execute_task(g_context, launcher);
        return legion_future<void>(result);
    }

    void get()
    {
        if (valid_)
            fut_.get_void_result();
    }
};

inline legion_future<void> make_ready_future()
{
    return legion_future<void>();   // default ctor ⇒ ready
}

// ===================================================================
//  Async launch helper – replaces hpx::async / std::async
// ===================================================================

template <typename F, typename... Args>
legion_future<void> async_launch(F&& f, Args&&... args)
{
    assert(g_runtime != nullptr);

    auto bound = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    int64_t cb_id = callback_registry::register_fn(
        [bound]() mutable { bound(); });

    Legion::TaskLauncher launcher(
        ASTM_CALLBACK_TASK_ID,
        Legion::TaskArgument(&cb_id, sizeof(int64_t)));

    Legion::Future result =
        g_runtime->execute_task(g_context, launcher);
    return legion_future<void>(result);
}

// -------------------------------------------------------------------
// Convenience macros consumed by the rest of the ASTM code and by
// the application-level .cpp files.
// -------------------------------------------------------------------
#define ASTM_FUTURE          astm::legion_future
#define ASTM_MAKE_READY_FUTURE astm::make_ready_future
#define ASTM_ASYNC           astm::async_launch

// ===================================================================
//  shared_var_base
// ===================================================================

struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;

    virtual void write(shared_var_base const&) = 0;

    virtual ASTM_LOCK lock() const = 0;

    virtual bool operator==(shared_var_base const&) const = 0;
};

// Forward declaration
struct transaction;

// ===================================================================
//  transaction_future
// ===================================================================

struct transaction_future
{
    typedef ASTM_FUTURE<void> future_type;

  private:
    transaction* trans_;
    future_type  fut_;

  public:
    transaction_future(transaction* trans)
      : trans_(trans)
      , fut_(ASTM_MAKE_READY_FUTURE())
    {}

    transaction_future(transaction& trans)
      : trans_(&trans)
      , fut_(ASTM_MAKE_READY_FUTURE())
    {}

    template <typename F>
    void then(F f);

    void get()
    {
        fut_.get();
    }
};

// ===================================================================
//  shared_var<T>
// ===================================================================

template <typename T>
struct shared_var : shared_var_base
{
    typedef ASTM_FUTURE<void> future_type;

    // ---------------------------------------------------------------
    //  local_var – transaction-local view of the shared variable
    // ---------------------------------------------------------------
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
    T             data_;
    mutable ASTM_MUTEX mtx_;

  public:
    future_type queue;

    shared_var()
      : data_(), mtx_(), queue(ASTM_MAKE_READY_FUTURE())
    {}

    shared_var(T const& t)
      : data_(t), mtx_(), queue(ASTM_MAKE_READY_FUTURE())
    {}

    shared_var(T&& t)
      : data_(t), mtx_(), queue(ASTM_MAKE_READY_FUTURE())
    {}

    shared_var(shared_var const& rhs)
      : data_(rhs.data_), mtx_(), queue(ASTM_MAKE_READY_FUTURE())
    {}

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

// ===================================================================
//  transaction
// ===================================================================

struct transaction
{
    std::list<
        std::pair<
            shared_var_base*                      // shared variable we read
          , std::shared_ptr<shared_var_base>       // value at time of read
        >
    > read_list;

    std::set<
        shared_var_base*                           // shared variable we wrote
    > write_set;

    std::list<
        std::pair<
            ASTM_FUTURE<void>*                     // future to chain onto (NULL ⇒ fire-and-forget)
          , ASTM_FUNCTION<void(transaction*)>      // the continuation
        >
    > async_list;

    std::map<
        shared_var_base*                           // shared variable
      , std::shared_ptr<shared_var_base>           // current local value
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
        // 1.) Obtain exclusive access to all variables (sorted order).
        std::list<ASTM_LOCK> locks;

        for ( std::pair<shared_var_base*, std::shared_ptr<shared_var_base> > const& var
            : variables)
        {
            assert(var.first != NULL);
            locks.push_back((*var.first).lock());
        }

        // 2.) Validate recorded reads against current values.
        for ( std::pair<shared_var_base*, std::shared_ptr<shared_var_base> > const& var
            : read_list)
        {
            assert(var.first != NULL);

            if (!((*var.first) == (*var.second)))
            {
                clear();
                return false;       // transaction aborted
            }
        }

        // 3.) Perform writes from our internal map.
        for (shared_var_base* var : write_set)
        {
            assert(var != NULL);

            auto it = variables.find(var);
            assert(it != variables.end());

            (*var).write((*(*it).second));
        }

        // 4.) Perform async / continuation operations.
        for ( std::pair<ASTM_FUTURE<void>*, ASTM_FUNCTION<void(transaction*)> >& op
            : async_list)
        {
            if (op.first == NULL)
                // fire-and-forget
                ASTM_ASYNC(op.second, this);
            else
                (*op.first) = (*op.first).then(std::bind(op.second, this));
        }

        // 5.) Release exclusive access (RAII).
        return true;
    }

    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != NULL);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base> > entry(var, 0);

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

// ===================================================================
//  Out-of-line template definitions
// ===================================================================

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
    trans_->then(&dynamic_cast<shared_var const*>(&trans_)->queue, f);
}

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_);
    trans_->then(&fut_, f);
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
