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
#include <cassert>
#include <memory>
#include <utility>

// ---------------------------------------------------------------------------
// Custom future wrapper that supports .then() chaining.
// Legion::Future does not expose .then(), and std::future also lacks it.
// The STM commit path (astm.hpp – commit_transaction) requires:
//     (*op.first) = (*op.first).then(std::bind(op.second, this));
// so we provide a thin wrapper around std::shared_future<void>.
// ---------------------------------------------------------------------------
namespace astm_detail {

template <typename T>
class chainable_future;

template <>
class chainable_future<void> {
    std::shared_future<void> fut_;

public:
    chainable_future() : fut_() {}

    chainable_future(std::future<void>&& f)
      : fut_(f.share()) {}

    chainable_future(std::shared_future<void> f)
      : fut_(std::move(f)) {}

    chainable_future(chainable_future&&) = default;
    chainable_future& operator=(chainable_future&&) = default;
    chainable_future(const chainable_future&) = default;
    chainable_future& operator=(const chainable_future&) = default;

    void get() {
        if (fut_.valid())
            fut_.get();
    }

    bool valid() const { return fut_.valid(); }

    // .then() – waits for the current future, then invokes f().
    // Returns a new chainable_future representing the continuation.
    template <typename F>
    chainable_future<void> then(F f) {
        auto shared = fut_;
        return chainable_future<void>(
            std::async(std::launch::async,
                [shared, func = std::move(f)]() mutable {
                    if (shared.valid())
                        shared.get();
                    func();
                }));
    }
};

// Replacement for ASTM_ASYNC – launches work asynchronously and returns a
// chainable_future<void>.  Within a Legion program the runtime's own threads
// coexist with application-spawned std::async threads.
template <typename F, typename... Args>
chainable_future<void> legion_async(F&& f, Args&&... args) {
    return chainable_future<void>(
        std::async(std::launch::async,
                   std::forward<F>(f),
                   std::forward<Args>(args)...));
}

// Replacement for ASTM_MAKE_READY_FUTURE – returns an already-satisfied
// chainable_future<void>.
inline chainable_future<void> make_ready_future() {
    std::promise<void> p;
    p.set_value();
    return chainable_future<void>(p.get_future());
}

} // namespace astm_detail

// ---- Primitive mappings used throughout the ASTM library ------------------
#define ASTM_MUTEX    std::mutex
#define ASTM_LOCK     std::unique_lock<std::mutex>
#define ASTM_FUTURE   astm_detail::chainable_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC    astm_detail::legion_async
// Note: defined as a callable so that ASTM_MAKE_READY_FUTURE() expands to
// astm_detail::make_ready_future(), matching the HPX pattern.
#define ASTM_MAKE_READY_FUTURE astm_detail::make_ready_future

#define ASTM_TEST   assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// ---------------------------------------------------------------------------
// Newer ASTM forward-include guard (kept for compatibility)
// ---------------------------------------------------------------------------
#if !defined(ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E)
#define ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E

#include "astm.hpp"

namespace astm
{
}

#endif
