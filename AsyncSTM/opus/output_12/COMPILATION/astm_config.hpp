////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// Legion execution model
#include "legion.h"

#include <future>
#include <thread>
#include <mutex>
#include <functional>
#include <memory>
#include <fstream>
#include <cassert>

namespace astm {

///////////////////////////////////////////////////////////////////////////////
// A future wrapper that supports .then() chaining.
//
// In the Legion execution model, program-level concurrency is managed through
// Legion tasks and logical regions. The STM's internal async operations use
// standard C++ concurrency primitives (threads, promises) to implement
// transactional semantics within or across Legion tasks sharing an address
// space.
///////////////////////////////////////////////////////////////////////////////
template<typename T>
struct legion_future;

template<>
struct legion_future<void> {
    std::shared_future<void> fut_;

    legion_future() : fut_() {}

    legion_future(std::future<void>&& f) : fut_(f.share()) {}

    legion_future(std::shared_future<void> const& f) : fut_(f) {}

    legion_future(const legion_future&) = default;
    legion_future(legion_future&&) = default;
    legion_future& operator=(const legion_future&) = default;
    legion_future& operator=(legion_future&&) = default;

    void get() {
        if (fut_.valid()) fut_.get();
    }

    // Chains a continuation after this future completes.
    // The continuation f is invoked with no arguments once the current
    // future is ready.  (std::bind callables with all parameters already
    // bound silently ignore any extra arguments, so this is compatible
    // with the HPX .then() calling convention.)
    template<typename F>
    legion_future<void> then(F f) {
        auto captured_fut = fut_;
        auto prom = std::make_shared<std::promise<void>>();
        std::future<void> new_fut = prom->get_future();

        std::thread([captured_fut, f, prom]() mutable {
            try {
                if (captured_fut.valid()) captured_fut.get();
                f();
                prom->set_value();
            } catch (...) {
                prom->set_exception(std::current_exception());
            }
        }).detach();

        return legion_future<void>(std::move(new_fut));
    }
};

///////////////////////////////////////////////////////////////////////////////
// Creates an immediately-ready future.
///////////////////////////////////////////////////////////////////////////////
inline legion_future<void> make_legion_ready_future() {
    std::promise<void> p;
    p.set_value();
    return legion_future<void>(p.get_future());
}

///////////////////////////////////////////////////////////////////////////////
// Launches an asynchronous operation and returns a legion_future.
//
// Uses std::thread (detached) + promise rather than std::async to avoid the
// blocking-destructor semantics of std::async, enabling true fire-and-forget
// when the returned future is discarded.
///////////////////////////////////////////////////////////////////////////////
template<typename F, typename... Args>
legion_future<void> legion_async(F&& f, Args&&... args) {
    auto prom = std::make_shared<std::promise<void>>();
    std::future<void> fut = prom->get_future();

    auto bound = std::make_shared<std::function<void()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::thread([prom, bound]() {
        try {
            (*bound)();
            prom->set_value();
        } catch (...) {
            prom->set_exception(std::current_exception());
        }
    }).detach();

    return legion_future<void>(std::move(fut));
}

} // namespace astm

// ---------------------------------------------------------------------------
// Macro definitions mapping ASTM primitives to Legion-compatible types
// ---------------------------------------------------------------------------

// Locking: std::mutex is used for the STM's optimistic-concurrency locks.
// Legion manages inter-task coherence via regions; these mutexes guard
// intra-address-space transactional state.
#define ASTM_MUTEX   std::mutex
#define ASTM_LOCK    std::unique_lock<std::mutex>

// Futures / async
#define ASTM_FUTURE              astm::legion_future
#define ASTM_FUNCTION            std::function
#define ASTM_ASYNC               astm::legion_async
#define ASTM_MAKE_READY_FUTURE   astm::make_legion_ready_future()

// Testing
#define ASTM_TEST   assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// ---------------------------------------------------------------------------
// Newer ASTM extension point
// ---------------------------------------------------------------------------
#if !defined(ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E)
#define ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E

#include "astm.hpp"

namespace astm
{
}

#endif
