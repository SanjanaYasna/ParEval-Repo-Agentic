////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// Translated to the Legion execution model.
//
// In Legion's task-based execution model:
// - Within a task, execution is sequential (no mutexes needed).
// - Cross-task synchronization is handled by the runtime via region requirements.
// - Async continuations within a task execute synchronously.
// - True parallelism is achieved through Legion task launches in calling code.

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94

#include "astm_config.hpp"

#include <list>
#include <set>
#include <map>
#include <vector>
#include <functional>
#include <memory>
#include <cassert>

// ============================================================
// Override concurrency macros for the Legion execution model.
// ASTM_TEST and ASTM_REPORT are kept from astm_config.hpp.
// ============================================================

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

// Provide fallback definitions for ASTM_TEST / ASTM_REPORT
// in case astm_config.hpp did not define them.
#ifndef ASTM_TEST
#include <cassert>
#define ASTM_TEST assert
#endif

#ifndef ASTM_REPORT
#define ASTM_REPORT 0
#endif

namespace astm
{

// ============================================================
// Legion-compatible future type.
//
// Within a Legion task execution is sequential, so then()
// executes the continuation immediately and returns a new
// (already-complete) future.
// ============================================================
template <typename T = void>
struct legion_future
{
    legion_future() = default;
    legion_future(legion_future const&) = default;
    legion_future(legion_future&&) = default;
    legion_future& operator=(legion_future const&) = default;
    legion_future& operator=(legion_future&&) = default;

    /// Execute continuation \a f synchronously, return a ready future.
    template <typename F>
    legion_future<void> then(F f)
    {
        f();
        return legion_future<void>();
    }

    void get() { /* already complete */ }
};

// ============================================================
// No-op synchronization primitives.
//
// Legion guarantees that a task body runs without concurrent
// interference on the regions it has mapped, so no locking is
// required within a task.  The noop types keep the commit code
// structurally identical to the original.
// ============================================================
struct noop_mutex {};

struct noop_lock
{
    noop_lock() = default;
    explicit noop_lock(noop_mutex const&) {}
    noop_lock(noop_lock&&) = default;
    noop_lock& operator=(noop_lock&&) = default;
};

// ============================================================
// Synchronous async helper.
//
// Inside astm.hpp the only user of ASTM_ASYNC is
// commit_transaction() for fire-and-forget / continuation
// operations.  True data-parallel work is launched as Legion
// tasks by the calling code.
// ============================================================
template <typename F, typename... Args>
legion_future<void> legion_async(F&& f, Args&&... args)
{
    f(std::forward<Args>(args)...);
    return legion_future<void>();
}

} // namespace astm

// ============================================================
// Macro bridge — map the generic ASTM_* names to Legion types.
// ============================================================
#define ASTM_MUTEX      astm::noop_mutex
#define ASTM_LOCK       astm::noop_lock
#define ASTM_FUTURE     astm::legion_future
#define ASTM_FUNCTION   std::function
#define ASTM_ASYNC      astm::legion_async
#define ASTM_MAKE_READY_FUTURE astm::legion_future<void>

namespace astm
{

// ============================================================
// shared_var_base — type-erased interface for transactional
// variables.
// ============================================================
struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;

    virtual void write(shared_var_base const&) = 0;

    virtual ASTM_LOCK lock() const = 0;

    virtual bool operator==(shared_var_base const&) const = 0;
};

struct transaction;

// ============================================================
// transaction_future — a future whose continuation is deferred
// until transaction commit.
// ============================================================
struct transaction_future
{
    typedef ASTM_FUTURE<void> future_type;

  private:
    transaction* trans_;
    future_type fut_;

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

// ============================================================
// shared_var<T> — a transactional variable of type T.
// ============================================================
template <typename T>
struct shared_var : shared_var_base
{
    typedef ASTM_FUTURE<void> future_type;

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

    // Locks (no-op in Legion) and returns a copy.
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

// ============================================================
// transaction — accumulates reads / writes / async actions and
// commits them atomically.
//
// Within a single Legion task there is no concurrent mutation,
// so commit will only fail when the user deliberately modifies
// a shared_var outside the transaction (the "retry" test).
// ============================================================
struct transaction
{
    std::list<
        std::pair<
            shared_var_base*                   // variable we're reading
          , std::shared_ptr<shared_var_base>   // snapshot of its value
        >
    > read_list;

    std::set<
        shared_var_base*                       // variable we're writing
    > write_set;

    std::list<
        std::pair<
            ASTM_FUTURE<void>*                           // target future (NULL → fire-and-forget)
          , ASTM_FUNCTION<void(transaction*)>            // deferred action
        >
    > async_list;

    std::map<
        shared_var_base*                       // variable
      , std::shared_ptr<shared_var_base>       // current local value
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
        //     (No-op in Legion — task already has exclusive access.)
        // 2.) Compare our recorded reads against the current values.
        // 3.) Perform writes.
        // 4.) Perform async / continuation operations.
        // 5.) Release exclusive access (RAII — no-op).

        // 1.) Acquire no-op locks (keeps structure identical to original).
        std::list<ASTM_LOCK> locks;
        for ( std::pair<shared_var_base*, std::shared_ptr<shared_var_base> > const& var
            : variables)
        {
            assert(var.first != NULL);
            locks.push_back((*var.first).lock());
        }

        // 2.) Validate recorded reads.
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

        // 3.) Apply writes from the internal map to the actual shared variables.
        for (shared_var_base* var : write_set)
        {
            assert(var != NULL);

            auto it = variables.find(var);
            assert(it != variables.end());

            (*var).write((*(*it).second));
        }

        // 4.) Execute deferred async / continuation operations.
        for ( std::pair<ASTM_FUTURE<void>*, ASTM_FUNCTION<void(transaction*)> >& op
            : async_list)
        {
            if (op.first == NULL)
                // Fire-and-forget: execute synchronously within this task.
                ASTM_ASYNC(op.second, this);
            else
                // Continuation: execute synchronously and assign the resulting
                // ready future back to the target.
                (*op.first) = (*op.first).then(std::bind(op.second, this));
        }

        // 5.) Locks released via RAII (no-op).
        return true;
    }

    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != NULL);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base> > entry(var, 0);

        auto result = variables.insert(entry);

        if (result.second == true)
        {
            // First access — clone the current value.
            (*result.first).second.reset((*var).clone());

            // Record the read for validation at commit time.
            read_list.push_back(*result.first);

            return (*(*result.first).second);
        }
        else
            // Already in the local map.
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

// ============================================================
// Out-of-line member definitions for shared_var<T>::local_var
// ============================================================

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
