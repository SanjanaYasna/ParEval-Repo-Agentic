////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
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
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <cstring>

// -------------------------------------------------------------------------
// Macros for Legion backend
// -------------------------------------------------------------------------
#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<std::mutex>
#define ASTM_FUNCTION std::function
#define ASTM_TEST assert
#define ASTM_REPORT 0

namespace astm {

// Forward declarations
struct transaction;

// -------------------------------------------------------------------------
// Thread-local Legion runtime and context.
// Must be set at the start of every Legion task (including top-level).
// -------------------------------------------------------------------------
inline thread_local Legion::Runtime* tl_runtime  = nullptr;
inline thread_local Legion::Context  tl_context;

inline void set_legion_context(Legion::Runtime* rt, Legion::Context ctx)
{
    tl_runtime  = rt;
    tl_context  = ctx;
}

// -------------------------------------------------------------------------
// Task ID for the generic async wrapper task
// -------------------------------------------------------------------------
enum ASTMTaskIDs {
    ASTM_GENERIC_TASK_ID = 55555,
};

// -------------------------------------------------------------------------
// Function registry – allows passing arbitrary std::function<void()>
// through Legion task arguments by storing them in a process-global map
// keyed by a uint64_t id that is serialized into the TaskArgument.
// -------------------------------------------------------------------------
inline std::mutex& func_registry_mutex()
{
    static std::mutex mtx;
    return mtx;
}

inline std::map<uint64_t, std::function<void()>>& func_registry()
{
    static std::map<uint64_t, std::function<void()>> reg;
    return reg;
}

inline std::atomic<uint64_t>& next_func_id()
{
    static std::atomic<uint64_t> id{0};
    return id;
}

inline uint64_t register_function(std::function<void()> f)
{
    uint64_t id = next_func_id().fetch_add(1);
    std::lock_guard<std::mutex> lock(func_registry_mutex());
    func_registry()[id] = std::move(f);
    return id;
}

inline std::function<void()> retrieve_function(uint64_t id)
{
    std::lock_guard<std::mutex> lock(func_registry_mutex());
    auto& reg = func_registry();
    auto it = reg.find(id);
    assert(it != reg.end());
    auto f = std::move(it->second);
    reg.erase(it);
    return f;
}

// -------------------------------------------------------------------------
// Generic Legion task – looks up the registered callable and invokes it.
// -------------------------------------------------------------------------
inline void generic_task_impl(const Legion::Task* task,
                              const std::vector<Legion::PhysicalRegion>&,
                              Legion::Context ctx,
                              Legion::Runtime* runtime)
{
    // Propagate Legion context so that nested ASTM operations work.
    tl_runtime  = runtime;
    tl_context  = ctx;

    assert(task->arglen == sizeof(uint64_t));
    uint64_t func_id;
    std::memcpy(&func_id, task->args, sizeof(func_id));

    auto f = retrieve_function(func_id);
    f();
}

// -------------------------------------------------------------------------
// Call once *before* Runtime::start() to register the generic task.
// -------------------------------------------------------------------------
inline void preregister_astm_tasks()
{
    Legion::TaskVariantRegistrar registrar(ASTM_GENERIC_TASK_ID,
                                           "astm_generic_task");
    registrar.add_constraint(
        Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    Legion::Runtime::preregister_task_variant<generic_task_impl>(
        registrar, "astm_generic_task");
}

// -------------------------------------------------------------------------
// legion_future<T> – wraps Legion::Future and adds .then() / .get()
//
// A default-constructed legion_future is "immediately ready" (no actual
// Legion future behind it).  .then(F) launches a generic task that will
// execute F once the wrapped future (if any) is satisfied, using
// TaskLauncher::add_future to express the dependency.
// -------------------------------------------------------------------------
template <typename T = void>
class legion_future
{
    Legion::Future impl_;
    bool           has_future_;

  public:
    legion_future()
      : impl_(), has_future_(false)
    {}

    legion_future(Legion::Future f)
      : impl_(std::move(f)), has_future_(true)
    {}

    legion_future(const legion_future&)            = default;
    legion_future& operator=(const legion_future&) = default;
    legion_future(legion_future&&)                 = default;
    legion_future& operator=(legion_future&&)      = default;

    void get()
    {
        if (has_future_)
            impl_.get_void_result(true /*silence_warnings*/);
    }

    template <typename F>
    legion_future<void> then(F f)
    {
        std::function<void()> func(std::move(f));
        uint64_t id = register_function(std::move(func));

        Legion::TaskLauncher launcher(ASTM_GENERIC_TASK_ID,
                                      Legion::TaskArgument(&id, sizeof(id)));
        if (has_future_)
            launcher.add_future(impl_);          // ordering dependency

        assert(tl_runtime != nullptr);
        return legion_future<void>(
            tl_runtime->execute_task(tl_context, launcher));
    }
};

// -------------------------------------------------------------------------
// legion_async – variadic async launcher, returns legion_future<void>
// -------------------------------------------------------------------------
template <typename F, typename... Args>
legion_future<void> legion_async(F&& f, Args&&... args)
{
    std::function<void()> func =
        std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    uint64_t id = register_function(std::move(func));

    Legion::TaskLauncher launcher(ASTM_GENERIC_TASK_ID,
                                  Legion::TaskArgument(&id, sizeof(id)));
    assert(tl_runtime != nullptr);
    return legion_future<void>(
        tl_runtime->execute_task(tl_context, launcher));
}

// -------------------------------------------------------------------------
// make_ready_future – returns an already-satisfied future
// -------------------------------------------------------------------------
inline legion_future<void> make_ready_future()
{
    return legion_future<void>();       // default ctor → immediately ready
}

} // namespace astm

// -------------------------------------------------------------------------
// Macros consumed by the rest of the ASTM code
// -------------------------------------------------------------------------
#define ASTM_FUTURE            astm::legion_future
#define ASTM_ASYNC             astm::legion_async
#define ASTM_MAKE_READY_FUTURE astm::make_ready_future

// =========================================================================
//  Core ASTM types (transaction, shared_var, etc.)
// =========================================================================

namespace astm
{

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
      : data_(), mtx_(), queue(ASTM_MAKE_READY_FUTURE()) {}

    shared_var(T const& t)
      : data_(t), mtx_(), queue(ASTM_MAKE_READY_FUTURE()) {}

    shared_var(T&& t)
      : data_(t), mtx_(), queue(ASTM_MAKE_READY_FUTURE()) {}

    shared_var(shared_var const& rhs)
      : data_(rhs.data_), mtx_(), queue(ASTM_MAKE_READY_FUTURE()) {}

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

struct transaction
{
    std::list<
        std::pair<
            shared_var_base*                    // The shared variable we're reading from
          , std::shared_ptr<shared_var_base>    // The value we read
        >
    > read_list;

    std::set<
        shared_var_base*                        // The shared variable we're writing to
    > write_set;

    std::list<
        std::pair<
            ASTM_FUTURE<void>*                  // The future we're writing to
                                                // (if NULL, fire-and-forget)
          , ASTM_FUNCTION<void(transaction*)>   // The async action to execute
        >
    > async_list;

    std::map<
        shared_var_base*                        // The shared variable
      , std::shared_ptr<shared_var_base>        // Current value of the variable
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
        // 1.) Obtain exclusive access to all the variables.
        std::list<ASTM_LOCK> locks;

        for ( std::pair<shared_var_base*, std::shared_ptr<shared_var_base> > const& var
            : variables)
        {
            assert(var.first != NULL);
            locks.push_back((*var.first).lock());
        }

        // 2.) Compare our recorded reads against the current values.
        for ( std::pair<shared_var_base*, std::shared_ptr<shared_var_base> > const& var
            : read_list)
        {
            assert(var.first != NULL);

            if (!((*var.first) == (*var.second)))
            {
                clear();
                return false;   // Transaction fails; locks released by RAII.
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

// -------------------------------------------------------------------------
// Out-of-line template definitions
// -------------------------------------------------------------------------

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
