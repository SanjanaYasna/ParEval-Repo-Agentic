////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Translated to the Legion execution model.
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_LEGION_DBA88345_57B8_4CC8_A574_D5F007250E94)
#define ASTM_LEGION_DBA88345_57B8_4CC8_A574_D5F007250E94

#include "legion.h"

#include <list>
#include <set>
#include <map>
#include <vector>
#include <functional>
#include <mutex>
#include <future>
#include <memory>
#include <thread>
#include <cassert>
#include <cstring>

// ============================================================================
// Compatibility macros for translated .cpp files
// ============================================================================
#define ASTM_TEST assert
#define ASTM_REPORT 0

namespace astm
{

// ============================================================================
// Legion integration: thread-local runtime/context and task registration
// ============================================================================

enum AstmTaskIDs
{
    ASTM_GENERIC_TASK_ID = 88800,
};

namespace detail
{
    // Thread-local Legion runtime and context.
    // Set these at the beginning of each Legion task body via legion_scope
    // so that async_launch can use Legion tasks when available.
    inline thread_local Legion::Runtime* tl_runtime = nullptr;
    inline thread_local Legion::Context  tl_ctx;
} // namespace detail

// RAII scope guard: stores runtime/context in thread-local storage for the
// duration of a Legion task body so that astm::async_launch can use Legion
// tasks transparently.
struct legion_scope
{
    legion_scope(Legion::Runtime* rt, Legion::Context ctx)
    {
        detail::tl_runtime = rt;
        detail::tl_ctx     = ctx;
    }
    ~legion_scope()
    {
        detail::tl_runtime = nullptr;
    }

    legion_scope(legion_scope const&)            = delete;
    legion_scope& operator=(legion_scope const&) = delete;
};

// Generic Legion task body: reads a heap-allocated std::function<void()>*
// from the task arguments, invokes it, and deletes it.
// NOTE: This only works when the task executes in the same address space
// that created the std::function (i.e., same node — required for STM anyway).
inline void astm_generic_task_impl(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& /*regions*/,
    Legion::Context ctx,
    Legion::Runtime* runtime)
{
    // Propagate runtime/context so nested async_launch calls work.
    detail::tl_runtime = runtime;
    detail::tl_ctx     = ctx;

    std::function<void()>* fn_ptr = nullptr;
    assert(task->arglen == sizeof(fn_ptr));
    std::memcpy(&fn_ptr, task->args, sizeof(fn_ptr));
    (*fn_ptr)();
    delete fn_ptr;
}

// Call this once before Legion::Runtime::start().
inline void register_astm_tasks()
{
    Legion::TaskVariantRegistrar registrar(ASTM_GENERIC_TASK_ID,
                                           "astm_generic_task");
    registrar.add_constraint(
        Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    Legion::Runtime::preregister_task_variant<astm_generic_task_impl>(
        registrar, "astm_generic_task");
}

// ============================================================================
// legion_future<void> — a future type with .then() chaining support
// ============================================================================

template <typename T = void>
class legion_future;

template <>
class legion_future<void>
{
    enum class Kind { READY, STD_FUTURE, LEGION_FUTURE };

    Kind                     kind_;
    std::shared_future<void> std_fut_;
    Legion::Future           legion_fut_;

  public:
    // Default: already-complete future.
    legion_future()
      : kind_(Kind::READY)
      , std_fut_()
      , legion_fut_()
    {}

    // From a std::future<void> (takes ownership).
    explicit legion_future(std::future<void>&& f)
      : kind_(Kind::STD_FUTURE)
      , std_fut_(f.share())
      , legion_fut_()
    {}

    // From a std::shared_future<void>.
    explicit legion_future(std::shared_future<void> const& f)
      : kind_(Kind::STD_FUTURE)
      , std_fut_(f)
      , legion_fut_()
    {}

    // From a Legion::Future.
    explicit legion_future(Legion::Future const& f)
      : kind_(Kind::LEGION_FUTURE)
      , std_fut_()
      , legion_fut_(f)
    {}

    legion_future(legion_future const&)            = default;
    legion_future(legion_future&&)                 = default;
    legion_future& operator=(legion_future const&) = default;
    legion_future& operator=(legion_future&&)      = default;

    // Block until the future is complete.
    void get()
    {
        switch (kind_)
        {
            case Kind::READY:
                break;
            case Kind::STD_FUTURE:
                std_fut_.get();
                break;
            case Kind::LEGION_FUTURE:
                legion_fut_.get_void_result();
                break;
        }
    }

    // Continuation: execute f when this future completes, return new future.
    // f is invoked with no arguments (extra arguments from HPX-style .then()
    // are simply not passed; callers in the original code used std::bind to
    // create nullary callables anyway).
    template <typename F>
    legion_future<void> then(F f)
    {
        switch (kind_)
        {
            case Kind::READY:
            {
                return legion_future<void>(
                    std::async(std::launch::async,
                               [f = std::move(f)]() mutable { f(); }));
            }
            case Kind::STD_FUTURE:
            {
                auto prev = std_fut_;
                return legion_future<void>(
                    std::async(std::launch::async,
                               [prev, f = std::move(f)]() mutable {
                                   prev.get();
                                   f();
                               }));
            }
            case Kind::LEGION_FUTURE:
            {
                auto lfut = legion_fut_;
                return legion_future<void>(
                    std::async(std::launch::async,
                               [lfut, f = std::move(f)]() mutable {
                                   lfut.get_void_result();
                                   f();
                               }));
            }
        }
        return legion_future<void>(); // unreachable
    }
};

// Create an already-complete future.
inline legion_future<void> make_ready_future()
{
    return legion_future<void>();
}

// Asynchronous launch: if a Legion runtime/context is available on this thread
// (via legion_scope), launches a Legion task; otherwise falls back to
// std::async.  Returns a legion_future<void>.
template <typename F, typename... Args>
legion_future<void> async_launch(F&& f, Args&&... args)
{
    if (detail::tl_runtime)
    {
        auto* fn = new std::function<void()>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        Legion::TaskLauncher launcher(
            ASTM_GENERIC_TASK_ID,
            Legion::TaskArgument(&fn, sizeof(fn)));
        Legion::Future lfut =
            detail::tl_runtime->execute_task(detail::tl_ctx, launcher);
        return legion_future<void>(lfut);
    }
    else
    {
        return legion_future<void>(
            std::async(std::launch::async,
                       std::forward<F>(f),
                       std::forward<Args>(args)...));
    }
}

// ============================================================================
// Convenience macros so that translated .cpp files can use the same names
// ============================================================================
#define ASTM_FUTURE  astm::legion_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC   astm::async_launch
#define ASTM_MAKE_READY_FUTURE astm::make_ready_future
#define ASTM_MUTEX   std::mutex
#define ASTM_LOCK    std::unique_lock<std::mutex>

// ============================================================================
// STM core types
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

struct transaction_future
{
    typedef legion_future<void> future_type;

  private:
    transaction* trans_;
    future_type  fut_;

  public:
    transaction_future(transaction* trans)
      : trans_(trans)
      , fut_(make_ready_future())
    {}

    transaction_future(transaction& trans)
      : trans_(&trans)
      , fut_(make_ready_future())
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
    typedef legion_future<void> future_type;

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
    mutable std::mutex mtx_;

  public:
    future_type queue;

    shared_var()
      : data_(), mtx_(), queue(make_ready_future())
    {}

    shared_var(T const& t)
      : data_(t), mtx_(), queue(make_ready_future())
    {}

    shared_var(T&& t)
      : data_(t), mtx_(), queue(make_ready_future())
    {}

    shared_var(shared_var const& rhs)
      : data_(rhs.data_), mtx_(), queue(make_ready_future())
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
            legion_future<void>*                  // The future we're writing to
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

    // Attempt to commit the transaction.
    // Returns true on success, false if a conflict was detected (caller retries).
    bool commit_transaction()
    {
        // Algorithm:
        //
        // 1.) Obtain exclusive access to all the variables.
        // 2.) Compare our recorded reads against the current values (fail if
        //     needed).
        // 3.) Perform writes, reading from our internal map.
        // 4.) Perform async operations.
        // 5.) Release exclusive access.

        // 1.) Obtain exclusive access to all the variables.
        std::list<std::unique_lock<std::mutex>> locks;

        // Variable map is sorted, so order of locking is sorted.
        for ( std::pair<shared_var_base*, std::shared_ptr<shared_var_base>> const& var
            : variables)
        {
            assert(var.first != NULL);
            locks.push_back((*var.first).lock());
        }

        // 2.) Compare our recorded reads against the current values (fail if
        //     needed).
        for ( std::pair<shared_var_base*, std::shared_ptr<shared_var_base>> const& var
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
        for ( std::pair<legion_future<void>*, std::function<void(transaction*)>>& op
            : async_list)
        {
            // If the future pointer is null, use fire-and-forget semantics.
            if (op.first == NULL)
                async_launch(op.second, this);
            else
                (*op.first) = (*op.first).then(std::bind(op.second, this));
        }

        // 5.) Release exclusive access.
        return true;
    }

    shared_var_base const& read(shared_var_base* var)
    {
        // Two cases:
        // * First access: variable not in internal state — perform clone.
        // * Subsequent access: variable already in internal state — return it.

        assert(var != NULL);

        // Attempt insertion with a placeholder value.
        std::pair<shared_var_base*, std::shared_ptr<shared_var_base>> entry(var, nullptr);

        auto result = variables.insert(entry);

        if (result.second == true)
        {
            // Insertion succeeded; first read of the variable.
            (*result.first).second.reset((*var).clone()); // Perform read.

            // Record the read operation.
            read_list.push_back(*result.first);

            return (*(*result.first).second);
        }
        else
        {
            // Insertion failed; variable is in internal state.
            return (*(*result.first).second);
        }
    }

    void write(shared_var_base* var, shared_var_base const& value)
    {
        // Two cases:
        // * First access: variable not in internal state.
        // * Subsequent access: variable already in internal state.

        assert(var != NULL);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base>>
            entry(var, std::shared_ptr<shared_var_base>(value.clone()));

        auto result = variables.insert(entry);

        if (result.second == true)
        {
            // Insertion succeeded; first write of the variable.
            write_set.insert(var);
        }
        else
        {
            // Insertion failed; update the internal state.
            (*result.first).second = entry.second; // Perform INTERNAL write.
            write_set.insert(var);
        }
    }

    // Schedule an async operation to run at commit time.
    // If fut is NULL, fire-and-forget semantics are used.
    void then(legion_future<void>* fut, std::function<void(transaction*)> F)
    {
        std::pair<legion_future<void>*, std::function<void(transaction*)>>
            entry(fut, F);
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
    trans_->then(&(dynamic_cast<shared_var*>(var_)->queue), f);
}

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_);
    trans_->then(&fut_, f);
}

} // namespace astm

#endif // ASTM_LEGION_DBA88345_57B8_4CC8_A574_D5F007250E94
