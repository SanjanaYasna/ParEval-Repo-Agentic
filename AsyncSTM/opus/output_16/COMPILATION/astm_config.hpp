////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

#define ASTM_LEGION

#include "legion.h"

#include <mutex>
#include <functional>
#include <cassert>
#include <fstream>
#include <map>
#include <atomic>
#include <cstring>

namespace astm {

// ---------------------------------------------------------------
// Task IDs for ASTM internal use
// ---------------------------------------------------------------
enum ASTMTaskIDs {
    ASTM_WORKER_TASK_ID = 5000,
};

// ---------------------------------------------------------------
// Thread-local Legion runtime and context storage.
// Must be set at the beginning of each Legion task via
// set_legion_context() so that ASTM_ASYNC / .then() can
// launch child tasks.
// ---------------------------------------------------------------
inline Legion::Runtime*& current_runtime() {
    static thread_local Legion::Runtime* rt = nullptr;
    return rt;
}

inline Legion::Context& current_context() {
    static thread_local Legion::Context ctx;
    return ctx;
}

inline void set_legion_context(Legion::Runtime* rt, Legion::Context c) {
    current_runtime() = rt;
    current_context() = c;
}

// ---------------------------------------------------------------
// Callable registry: maps unique IDs to std::function<void()>
// objects so that arbitrary callables can be dispatched through
// the single registered Legion worker task.
// ---------------------------------------------------------------
inline std::mutex& callable_registry_mutex() {
    static std::mutex mtx;
    return mtx;
}

inline std::map<size_t, std::function<void()>>& callable_registry() {
    static std::map<size_t, std::function<void()>> reg;
    return reg;
}

inline std::atomic<size_t>& next_callable_id() {
    static std::atomic<size_t> id{0};
    return id;
}

inline size_t register_callable(std::function<void()> fn) {
    size_t id = next_callable_id().fetch_add(1);
    {
        std::lock_guard<std::mutex> lock(callable_registry_mutex());
        callable_registry()[id] = std::move(fn);
    }
    return id;
}

inline std::function<void()> retrieve_callable(size_t id) {
    std::lock_guard<std::mutex> lock(callable_registry_mutex());
    auto it = callable_registry().find(id);
    assert(it != callable_registry().end());
    std::function<void()> fn = std::move(it->second);
    callable_registry().erase(it);
    return fn;
}

// ---------------------------------------------------------------
// Generic worker task: retrieves a callable from the registry
// by ID (passed via TaskArgument) and executes it.
// ---------------------------------------------------------------
inline void astm_worker_task_impl(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& /*regions*/,
    Legion::Context ctx, Legion::Runtime* runtime)
{
    // Propagate runtime/context so nested ASTM_ASYNC / .then() work
    set_legion_context(runtime, ctx);

    size_t callable_id;
    assert(task->arglen == sizeof(size_t));
    std::memcpy(&callable_id, task->args, sizeof(size_t));

    auto fn = retrieve_callable(callable_id);
    fn();
}

// ---------------------------------------------------------------
// Future wrapper providing .get() and .then() over Legion::Future
//
// Legion::Future is untyped; get_result<T>() / get_void_result()
// interpret the stored bytes.  This wrapper adds the .then()
// continuation pattern used by ASTM by launching a dependent
// Legion worker task.
// ---------------------------------------------------------------
template<typename T>
struct astm_future {
    Legion::Future fut;
    bool is_ready;

    astm_future() : fut(), is_ready(true) {}
    astm_future(Legion::Future f) : fut(f), is_ready(false) {}

    astm_future(const astm_future&) = default;
    astm_future& operator=(const astm_future&) = default;
    astm_future(astm_future&&) noexcept = default;
    astm_future& operator=(astm_future&&) noexcept = default;

    T get() {
        if (!is_ready) {
            is_ready = true;
            return fut.get_result<T>();
        }
        return T{};
    }

    template<typename F>
    astm_future<void> then(F f) {
        // Wrap the callable so it takes no arguments.
        // In ASTM usage the callable is always a fully-bound
        // std::bind result (void()), so calling f() is correct.
        std::function<void()> wrapper =
            [f_copy = std::move(f)]() mutable { f_copy(); };

        size_t callable_id = register_callable(std::move(wrapper));

        Legion::Runtime* rt = current_runtime();
        Legion::Context ctx = current_context();
        assert(rt != nullptr);

        Legion::TaskLauncher launcher(
            ASTM_WORKER_TASK_ID,
            Legion::TaskArgument(&callable_id, sizeof(size_t)));

        // Express the data dependency: the worker task will not
        // start until the predecessor future is ready.
        if (!is_ready) {
            launcher.add_future(fut);
        }

        Legion::Future new_fut = rt->execute_task(ctx, launcher);
        return astm_future<void>(new_fut);
    }
};

// Specialization for void
template<>
struct astm_future<void> {
    Legion::Future fut;
    bool is_ready;

    astm_future() : fut(), is_ready(true) {}
    astm_future(Legion::Future f) : fut(f), is_ready(false) {}

    astm_future(const astm_future&) = default;
    astm_future& operator=(const astm_future&) = default;
    astm_future(astm_future&&) noexcept = default;
    astm_future& operator=(astm_future&&) noexcept = default;

    void get() {
        if (!is_ready) {
            fut.get_void_result();
            is_ready = true;
        }
    }

    template<typename F>
    astm_future<void> then(F f) {
        std::function<void()> wrapper =
            [f_copy = std::move(f)]() mutable { f_copy(); };

        size_t callable_id = register_callable(std::move(wrapper));

        Legion::Runtime* rt = current_runtime();
        Legion::Context ctx = current_context();
        assert(rt != nullptr);

        Legion::TaskLauncher launcher(
            ASTM_WORKER_TASK_ID,
            Legion::TaskArgument(&callable_id, sizeof(size_t)));

        if (!is_ready) {
            launcher.add_future(fut);
        }

        Legion::Future new_fut = rt->execute_task(ctx, launcher);
        return astm_future<void>(new_fut);
    }
};

// ---------------------------------------------------------------
// Async launch: stores callable+args in the registry, launches
// a Legion worker task, and returns a future for the result.
// ---------------------------------------------------------------
template<typename F, typename... Args>
astm_future<void> astm_async(F&& f, Args&&... args) {
    std::function<void()> fn;
    if constexpr (sizeof...(Args) == 0) {
        // Callable is already fully bound (e.g. from std::bind)
        fn = [f_copy = std::decay_t<F>(std::forward<F>(f))]() mutable {
            f_copy();
        };
    } else {
        // Bind the arguments to produce a void() callable
        auto bound = std::bind(std::forward<F>(f),
                               std::forward<Args>(args)...);
        fn = [bound = std::move(bound)]() mutable { bound(); };
    }

    size_t callable_id = register_callable(std::move(fn));

    Legion::Runtime* rt = current_runtime();
    Legion::Context ctx = current_context();
    assert(rt != nullptr);

    Legion::TaskLauncher launcher(
        ASTM_WORKER_TASK_ID,
        Legion::TaskArgument(&callable_id, sizeof(size_t)));

    Legion::Future new_fut = rt->execute_task(ctx, launcher);
    return astm_future<void>(new_fut);
}

// ---------------------------------------------------------------
// Create an already-completed (ready) future
// ---------------------------------------------------------------
inline astm_future<void> make_ready_future() {
    return astm_future<void>();
}

// ---------------------------------------------------------------
// Register ASTM internal Legion tasks.
// Must be called before Legion::Runtime::start().
// ---------------------------------------------------------------
inline void register_astm_tasks() {
    Legion::TaskVariantRegistrar registrar(ASTM_WORKER_TASK_ID,
                                          "astm_worker");
    registrar.add_constraint(
        Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    Legion::Runtime::preregister_task_variant<astm_worker_task_impl>(
        registrar, "astm_worker");
}

} // namespace astm

// ---------------------------------------------------------------
// ASTM Portability Macros — Legion Execution Model
// ---------------------------------------------------------------

// Mutual exclusion for the STM commit protocol (intra-node sync)
#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<std::mutex>

// Futures and async — backed by Legion tasks and Legion::Future
#define ASTM_FUTURE astm::astm_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC astm::astm_async
#define ASTM_MAKE_READY_FUTURE astm::make_ready_future()

// Testing
#define ASTM_TEST assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// ---------------------------------------------------------------
// Newer ASTM section (preserved from original)
// ---------------------------------------------------------------
#if !defined(ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E)
#define ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E

#include "astm.hpp"

namespace astm
{
}

#endif
