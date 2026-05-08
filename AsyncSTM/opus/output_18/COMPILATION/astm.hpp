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
#include <memory>
#include <mutex>
#include <future>
#include <cassert>
#include <fstream>

// ---- Legion-specific ASTM configuration macros ----
#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<std::mutex>
#define ASTM_FUNCTION std::function
#define ASTM_TEST assert
#define ASTM_REPORT 0

namespace astm
{

// ---------------------------------------------------------------------------
// Thread-local Legion runtime context.
// Each Legion task entry point must call set_legion_context().
// ---------------------------------------------------------------------------
inline thread_local Legion::Runtime* tl_runtime  = nullptr;
inline thread_local Legion::Context  tl_context  = {};

inline void set_legion_context(Legion::Runtime* rt, Legion::Context ctx)
{
    tl_runtime = rt;
    tl_context = ctx;
}

// ---------------------------------------------------------------------------
// Chainable future – a lightweight future that supports .then() chaining
// by accumulating work and executing it synchronously on .get().
// This replaces both hpx::future and std::future in the ASTM internals.
// ---------------------------------------------------------------------------
struct chainable_future
{
    std::shared_ptr<std::function<void()>> chain_;

    chainable_future()
      : chain_(std::make_shared<std::function<void()>>([](){}))
    {}

    // Run the chain to completion.
    void get()
    {
        if (chain_) (*chain_)();
        // Reset so repeated get() is safe.
        chain_ = std::make_shared<std::function<void()>>([](){});
    }

    // Append work after the current chain; returns a new future that
    // represents the combined sequence.
    template <typename F>
    chainable_future then(F f)
    {
        auto prev = chain_;
        chainable_future result;
        result.chain_ = std::make_shared<std::function<void()>>(
            [prev, f]() mutable {
                if (prev) (*prev)();
                f();
            });
        return result;
    }
};

// ---------------------------------------------------------------------------
// Async future – wraps std::future<void> so that ASTM_ASYNC callers can
// collect futures and .get() them, while still providing real concurrency.
// ---------------------------------------------------------------------------
struct async_future
{
    std::shared_future<void> fut_;

    async_future() : fut_() {}
    async_future(std::future<void>&& f) : fut_(f.share()) {}
    async_future(std::shared_future<void> f) : fut_(f) {}

    void get() { if (fut_.valid()) fut_.get(); }
};

// ---------------------------------------------------------------------------
// ASTM_FUTURE<void>  – used inside the STM for chaining.
// ASTM_ASYNC_FUTURE  – used by callers that launch real concurrent work.
// ---------------------------------------------------------------------------
#define ASTM_FUTURE astm::chainable_future

inline chainable_future ASTM_MAKE_READY_FUTURE()
{
    return chainable_future();
}

// Launch a callable asynchronously and return an async_future.
template <typename F, typename... Args>
async_future astm_async(F&& f, Args&&... args)
{
    return async_future(
        std::async(std::launch::async,
                   std::forward<F>(f),
                   std::forward<Args>(args)...));
}

#define ASTM_ASYNC astm::astm_async

// ---------------------------------------------------------------------------
// shared_var_base – type-erased interface for transactional variables.
// ---------------------------------------------------------------------------
struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;

    virtual void write(shared_var_base const&) = 0;

    virtual ASTM_LOCK lock() const = 0;

    virtual bool operator==(shared_var_base const&) const = 0;
};

struct transaction;

// ---------------------------------------------------------------------------
// transaction_future – allows scheduling deferred work inside a transaction.
// ---------------------------------------------------------------------------
struct transaction_future
{
    typedef chainable_future future_type;

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

// ---------------------------------------------------------------------------
// shared_var<T> – a transactional variable holding a value of type T.
// ---------------------------------------------------------------------------
template <typename T>
struct shared_var : shared_var_base
{
    typedef chainable_future future_type;

    struct local_var
    {
      private:
        transaction* trans_;
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
    mutable ASTM_MUTEX mtx_;

  public:
    future_type queue;

    shared_var() : data_(), mtx_(), queue(ASTM_MAKE_READY_FUTURE()) {}

    shared_var(T const& t) : data_(t), mtx_(), queue(ASTM_MAKE_READY_FUTURE()) {}

    shared_var(T&& t) : data_(t), mtx_(), queue(ASTM_MAKE_READY_FUTURE()) {}

    shared_var(shared_var const& rhs) : data_(rhs.data_), mtx_(), queue(ASTM_MAKE_READY_FUTURE()) {}

    ~shared_var() { }

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

// ---------------------------------------------------------------------------
// transaction – manages an optimistic STM transaction.
// ---------------------------------------------------------------------------
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
            chainable_future*             // The future we're writing to
                                          // (if NULL, fire-and-forget semantics)
          , ASTM_FUNCTION<void(transaction*)> // The async action to execute
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

    // Commit the transaction.  Returns true on success, false on conflict.
    bool commit_transaction()
    {
        // Algorithm:
        //
        // 1.) Obtain exclusive access to all the variables.
        // 2.) Compare our recorded reads against the current values (fail if needed).
        // 3.) Perform writes, reading from our internal map.
        // 4.) Perform async operations.
        // 5.) Release exclusive access.

        // 1.) Obtain exclusive access to all the variables.
        std::list<ASTM_LOCK> locks;

        // Variable map is sorted, so order of locking is sorted.
        for ( std::pair<shared_var_base*, std::shared_ptr<shared_var_base> > const& var
            : variables)
        {
            assert(var.first != NULL);

            locks.push_back((*var.first).lock());
        }

        // 2.) Compare our recorded reads against the current values (fail if needed).
        for ( std::pair<shared_var_base*, std::shared_ptr<shared_var_base> > const& var
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
        for ( std::pair<chainable_future*, ASTM_FUNCTION<void(transaction*)> >& op
            : async_list)
        {
            // If the future pointer is NULL, use fire-and-forget (run inline).
            if (op.first == NULL)
            {
                op.second(this);
            }
            else
            {
                // Chain work onto the existing future.
                auto bound = std::bind(op.second, this);
                (*op.first) = (*op.first).then(bound);
            }
        }

        // 5.) Release exclusive access (RAII – locks destroyed here).
        return true;
    }

    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != NULL);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base> > entry(var, 0);

        auto result = variables.insert(entry);

        if (result.second == true)
        {
            // Insertion succeeded; this is the first read of the variable.
            (*result.first).second.reset((*var).clone()); // Perform read.

            // Record the read operation.
            read_list.push_back(*result.first);

            return (*(*result.first).second);
        }
        else
            // Insertion failed; the variable is already in the internal state.
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
            // Insertion succeeded; this is the first access of the variable.
            write_set.insert(var);
        }
        else
        {
            // Insertion failed; the variable is already in the internal state.
            (*result.first).second = entry.second; // Perform INTERNAL write.

            write_set.insert(var);
        }
    }

    // If fut is NULL, then fire-and-forget semantics are used.
    void then(chainable_future* fut, ASTM_FUNCTION<void(transaction*)> F)
    {
        std::pair<chainable_future*, ASTM_FUNCTION<void(transaction*)> > entry(fut, F);
        async_list.push_back(entry);
    }
};

// ---------------------------------------------------------------------------
// shared_var<T>::local_var   out-of-line definitions
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Legion task registration helpers.
//
// Usage:
//   1. Define your task body with the signature:
//        void my_task(const Legion::Task*, ..., Legion::Context, Legion::Runtime*);
//   2. Register it before Runtime::start():
//        Legion::Runtime::register_task<my_task>(MY_TASK_ID);
//   3. Inside a Legion task, launch work:
//        astm::legion_launch(runtime, ctx, MY_TASK_ID);
// ---------------------------------------------------------------------------

// Launch a void Legion task and return an async_future that becomes ready
// when the task completes.
inline async_future legion_launch(Legion::Runtime* runtime,
                                  Legion::Context  ctx,
                                  Legion::TaskID   tid)
{
    Legion::TaskLauncher launcher(tid, Legion::TaskArgument(NULL, 0));
    Legion::Future f = runtime->execute_task(ctx, launcher);
    // Wrap the Legion::Future into an async_future via a blocking helper.
    return async_future(
        std::async(std::launch::deferred, [f]() mutable { f.get_void_result(); }));
}

// Launch a void Legion task with a serialisable argument pod.
template <typename ArgT>
inline async_future legion_launch(Legion::Runtime* runtime,
                                  Legion::Context  ctx,
                                  Legion::TaskID   tid,
                                  ArgT const&      arg)
{
    Legion::TaskLauncher launcher(
        tid,
        Legion::TaskArgument(&arg, sizeof(ArgT)));
    Legion::Future f = runtime->execute_task(ctx, launcher);
    return async_future(
        std::async(std::launch::deferred, [f]() mutable { f.get_void_result(); }));
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
