////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94

#include "astm_config.hpp"

// Undefine any macros from astm_config.hpp that conflict with Legion definitions
#ifdef ASTM_MUTEX
#undef ASTM_MUTEX
#endif
#ifdef ASTM_LOCK
#undef ASTM_LOCK
#endif
#ifdef ASTM_FUTURE
#undef ASTM_FUTURE
#endif
#ifdef ASTM_FUNCTION
#undef ASTM_FUNCTION
#endif
#ifdef ASTM_ASYNC
#undef ASTM_ASYNC
#endif
#ifdef ASTM_MAKE_READY_FUTURE
#undef ASTM_MAKE_READY_FUTURE
#endif
#ifdef ASTM_TEST
#undef ASTM_TEST
#endif
#ifdef ASTM_REPORT
#undef ASTM_REPORT
#endif

#include "legion.h"

#include <list>
#include <set>
#include <map>
#include <vector>
#include <functional>
#include <memory>
#include <cassert>
#include <mutex>
#include <cstring>
#include <fstream>

#define ASTM_TEST assert
#define ASTM_REPORT 0

namespace astm
{

// ============================================================================
// Legion context thread-local storage
// ============================================================================
// Each Legion task runs on a thread; store Runtime*/Context per thread so that
// ASTM helpers (shared_var, transaction, async) can access them.

inline Legion::Runtime*& tls_runtime()
{
    static thread_local Legion::Runtime* rt = nullptr;
    return rt;
}

inline Legion::Context& tls_context()
{
    static thread_local Legion::Context ctx;
    return ctx;
}

inline void set_legion_context(Legion::Runtime* rt, Legion::Context ctx)
{
    tls_runtime() = rt;
    tls_context() = ctx;
}

// ============================================================================
// Task IDs
// ============================================================================

enum ASTMTaskIDs
{
    ASTM_GENERIC_TASK_ID = 50000,
};

// ============================================================================
// Generic wrapper task – executes a heap-allocated std::function<void()>
// ============================================================================

inline void astm_generic_task_impl(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& /*regions*/,
    Legion::Context ctx,
    Legion::Runtime* runtime)
{
    set_legion_context(runtime, ctx);
    std::function<void()>* func = nullptr;
    assert(task->arglen == sizeof(func));
    std::memcpy(&func, task->args, sizeof(func));
    (*func)();
    delete func;
}

// Call once before Legion::Runtime::start()
inline void register_astm_tasks()
{
    Legion::TaskVariantRegistrar registrar(ASTM_GENERIC_TASK_ID,
                                           "astm_generic_task");
    registrar.add_constraint(
        Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    registrar.set_leaf(true);
    Legion::Runtime::preregister_task_variant<astm_generic_task_impl>(
        registrar, "astm_generic_task");
}

// ============================================================================
// future wrapper  –  wraps Legion::Future<void> and adds .then() / .get()
// ============================================================================

template <typename T = void>
struct astm_future;

template <>
struct astm_future<void>
{
    std::shared_ptr<Legion::Future> legion_fut_;
    bool ready_;

    // Default-construct as "ready" (equivalent to make_ready_future)
    astm_future() : legion_fut_(), ready_(true) {}

    // Construct from a real Legion future
    explicit astm_future(Legion::Future f)
      : legion_fut_(std::make_shared<Legion::Future>(std::move(f)))
      , ready_(false)
    {}

    // Copy / move
    astm_future(astm_future const& o) = default;
    astm_future(astm_future&& o) = default;
    astm_future& operator=(astm_future const& o) = default;
    astm_future& operator=(astm_future&& o) = default;

    void get()
    {
        if (!ready_ && legion_fut_)
        {
            legion_fut_->get_void_result();
            ready_ = true;
        }
    }

    // .then(f): wait for this future, then launch f as a child task.
    // Returns a new future representing f's completion.
    template <typename F>
    astm_future<void> then(F f);
};

// ============================================================================
// make_ready_future
// ============================================================================

inline astm_future<void> make_ready_future()
{
    return astm_future<void>(); // default ctor → ready
}

// ============================================================================
// Async launcher – packs a callable into a Legion child task
// ============================================================================

template <typename F, typename... Args>
astm_future<void> astm_async_impl(F&& f, Args&&... args)
{
    // Bind all arguments to produce a void() callable, heap-allocate it.
    auto* func = new std::function<void()>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    Legion::Runtime* rt  = tls_runtime();
    Legion::Context   ctx = tls_context();

    Legion::TaskLauncher launcher(
        ASTM_GENERIC_TASK_ID,
        Legion::TaskArgument(&func, sizeof(func)));

    return astm_future<void>(rt->execute_task(ctx, launcher));
}

// astm_future<void>::then implementation (needs astm_async_impl)
template <typename F>
astm_future<void> astm_future<void>::then(F f)
{
    get(); // wait for the current future to complete
    return astm_async_impl(std::move(f));
}

// ============================================================================
// Compatibility macros used by calling code
// ============================================================================

#define ASTM_MUTEX    std::mutex
#define ASTM_LOCK     std::unique_lock<std::mutex>
#define ASTM_FUTURE   astm::astm_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC    astm::astm_async_impl
#define ASTM_MAKE_READY_FUTURE astm::make_ready_future

// ============================================================================
// shared_var_base
// ============================================================================

struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;

    virtual void write(shared_var_base const&) = 0;

    virtual std::unique_lock<std::mutex> lock() const = 0;

    virtual bool operator==(shared_var_base const&) const = 0;
};

struct transaction;

// ============================================================================
// transaction_future
// ============================================================================

struct transaction_future
{
    typedef astm_future<void> future_type;

  private:
    transaction* trans_;
    future_type  fut_;

  public:
    transaction_future(transaction* trans)
      : trans_(trans)
      , fut_()          // ready future
    {}

    transaction_future(transaction& trans)
      : trans_(&trans)
      , fut_()
    {}

    template <typename F>
    void then(F f);

    void get()
    {
        fut_.get();
    }
};

// ============================================================================
// shared_var<T>
// ============================================================================

template <typename T>
struct shared_var : shared_var_base
{
    typedef astm_future<void> future_type;

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
    T data_;
    mutable std::mutex mtx_;

  public:
    future_type queue;

    shared_var()
      : data_(), mtx_(), queue()
    {}

    shared_var(T const& t)
      : data_(t), mtx_(), queue()
    {}

    shared_var(T&& t)
      : data_(t), mtx_(), queue()
    {}

    shared_var(shared_var const& rhs)
      : data_(rhs.data_), mtx_(), queue()
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

    std::unique_lock<std::mutex> lock() const
    {
        return std::unique_lock<std::mutex>(mtx_);
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

// ============================================================================
// transaction
// ============================================================================

struct transaction
{
    std::list<
        std::pair<
            // The shared variable we're reading from
            shared_var_base*
            // The value we read from the variable
          , std::shared_ptr<shared_var_base>
        >
    > read_list;

    std::set<
        shared_var_base* // The shared variable we're writing to
    > write_set;

    std::list<
        std::pair<
            astm_future<void>*                    // The future we're writing to
                                                  // (if NULL, fire-and-forget)
          , std::function<void(transaction*)>      // The async action to execute
        >
    > async_list;

    std::map<
        shared_var_base* // The shared variable we're reading from
      , std::shared_ptr<shared_var_base> // Current value of the variable
    > variables;

    void clear()
    {
        read_list.clear();
        write_set.clear();
        async_list.clear();
        variables.clear();
    }

    // This should possibly be called RAII-style from the destructor.
    bool commit_transaction()
    {
        // Algorithm:
        //
        // 1.) Obtain exclusive access to all the variables.
        // 2.) Compare our recorded reads against the current values
        //     (and fail if needed).
        // 3.) Perform writes, reading from our internal map.
        // 4.) Perform async operations.
        // 5.) Release exclusive access.

        // 1.) Obtain exclusive access to all the variables.
        std::list<std::unique_lock<std::mutex>> locks;

        // Variable map is sorted, so order of locking is sorted.
        for (std::pair<shared_var_base*, std::shared_ptr<shared_var_base>> const& var
            : variables)
        {
            assert(var.first != NULL);
            locks.push_back((*var.first).lock());
        }

        // 2.) Compare our recorded reads against the current values
        //     (and fail if needed).
        for (std::pair<shared_var_base*, std::shared_ptr<shared_var_base>> const& var
            : read_list)
        {
            assert(var.first != NULL);

            if (!((*var.first) == (*var.second))) // Perform read operation.
            {
                clear();

                // Transaction fails; exclusive access is released by RAII.
                return false;
            }
        }

        // 3.) Perform writes, reading from our internal map.
        for (shared_var_base* var : write_set)
        {
            assert(var != NULL);

            auto it = variables.find(var);

            assert(it != variables.end());

            (*var).write((*(*it).second)); // Perform write operation.
        }

        // 4.) Perform async operations.
        for (std::pair<astm_future<void>*,
                       std::function<void(transaction*)>>& op
            : async_list)
        {
            // If the future pointer is NULL, use fire-and-forget semantics.
            if (op.first == NULL)
                astm_async_impl(op.second, this);
            else
                (*op.first) = (*op.first).then(std::bind(op.second, this));
        }

        // 5.) Release exclusive access.
        return true;
    }

    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != NULL);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base>>
            entry(var, 0);

        auto result = variables.insert(entry);

        if (result.second == true)
        {
            (*result.first).second.reset((*var).clone()); // Perform read.
            read_list.push_back(*result.first);
            return (*(*result.first).second);
        }
        else
            return (*(*result.first).second);
    }

    void write(shared_var_base* var, shared_var_base const& value)
    {
        assert(var != NULL);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base>>
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

    // If fut is NULL, then fire-and-forget semantics are used.
    void then(astm_future<void>* fut,
              std::function<void(transaction*)> F)
    {
        std::pair<astm_future<void>*,
                  std::function<void(transaction*)>> entry(fut, F);
        async_list.push_back(entry);
    }
};

// ============================================================================
// shared_var<T>::local_var out-of-line definitions
// ============================================================================

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

// ============================================================================
// transaction_future::then
// ============================================================================

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_);
    trans_->then(&fut_, f);
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
