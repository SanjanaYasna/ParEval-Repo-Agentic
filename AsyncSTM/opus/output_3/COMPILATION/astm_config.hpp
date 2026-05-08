////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

#include "legion.h"

#include <future>
#include <thread>
#include <mutex>
#include <functional>
#include <memory>
#include <cassert>

namespace astm {
namespace detail {

// A future wrapper supporting .then() chaining.
// Neither std::future nor Legion::Future provide .then(), so we build
// this on std::shared_future and std::async to support the ASTM API
// requirement for continuation-based async operations.
template <typename T>
class chainable_future;

template <>
class chainable_future<void> {
    std::shared_future<void> fut_;
public:
    chainable_future() : fut_() {}
    chainable_future(std::future<void>&& f) : fut_(f.share()) {}
    chainable_future(std::shared_future<void> f) : fut_(std::move(f)) {}
    chainable_future(const chainable_future&) = default;
    chainable_future(chainable_future&&) = default;
    chainable_future& operator=(const chainable_future&) = default;
    chainable_future& operator=(chainable_future&&) = default;
    ~chainable_future() = default;

    // Chain a continuation: waits for the current future to complete, then
    // executes f, returning a new future representing completion of f.
    template <typename F>
    chainable_future<void> then(F&& f) {
        auto captured = fut_;
        std::future<void> new_fut = std::async(std::launch::async,
            [captured,
             func = std::function<void()>(std::forward<F>(f))]() mutable {
                if (captured.valid())
                    captured.get();
                func();
            });
        return chainable_future<void>(std::move(new_fut));
    }

    void get() {
        if (fut_.valid())
            fut_.get();
    }

    bool valid() const { return fut_.valid(); }
};

// Create an immediately-ready future (analogous to hpx::make_ready_future).
inline chainable_future<void> make_ready_future() {
    std::promise<void> p;
    p.set_value();
    return chainable_future<void>(p.get_future());
}

// Launch an asynchronous operation, returning a chainable_future.
template <typename F, typename... Args>
chainable_future<void> async_launch(F&& f, Args&&... args) {
    return chainable_future<void>(
        std::async(std::launch::async,
                   std::forward<F>(f),
                   std::forward<Args>(args)...));
}

} // namespace detail
} // namespace astm

// Mutex and lock types: std::mutex is used for intra-node STM commit
// synchronization, which remains valid within Legion task execution.
#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<std::mutex>

// Future, function, and async primitives: we use the chainable_future
// wrapper to provide .then() support required by the ASTM transaction
// commit and async chaining logic.
#define ASTM_FUTURE astm::detail::chainable_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC astm::detail::async_launch
#define ASTM_MAKE_READY_FUTURE astm::detail::make_ready_future

#define ASTM_TEST assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// Newer ASTM extension point
#if !defined(ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E)
#define ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E

#include "astm.hpp"

namespace astm
{
}

#endif
