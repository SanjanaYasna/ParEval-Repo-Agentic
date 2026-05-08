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

// In the Legion execution model, the runtime manages coarse-grained task
// parallelism and data placement via logical regions and the default mapper.
// For the internal STM commit protocol (fine-grained locking of shared
// variables), we use std:: synchronization primitives since Legion does not
// replace intra-node mutexes.

#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<std::mutex>

// ---------------------------------------------------------------------------
// Legion::Future does not natively support .then() continuation chaining
// that the ASTM transaction framework requires.  We provide a lightweight
// wrapper around std::shared_future that adds .then() support while remaining
// compatible with the rest of the ASTM API.
// ---------------------------------------------------------------------------

namespace astm {
namespace detail {

template <typename T>
class legion_future;

template <>
class legion_future<void> {
    std::shared_future<void> impl_;
public:
    legion_future() : impl_() {}
    legion_future(std::future<void>&& f) : impl_(f.share()) {}
    legion_future(std::shared_future<void> f) : impl_(std::move(f)) {}
    legion_future(legion_future&&) = default;
    legion_future(const legion_future&) = default;
    legion_future& operator=(legion_future&&) = default;
    legion_future& operator=(const legion_future&) = default;

    // Continuation support: schedules f to run after this future resolves.
    // Returns a new future representing completion of f.
    template <typename F>
    legion_future<void> then(F&& f) {
        // Copy the shared state so the lambda owns an independent handle.
        auto shared = impl_;
        std::future<void> fut = std::async(std::launch::async,
            [shared, func = std::forward<F>(f)]() mutable {
                if (shared.valid()) shared.wait();
                func();
            });
        return legion_future<void>(std::move(fut));
    }

    void get() {
        if (impl_.valid()) impl_.get();
    }

    bool valid() const { return impl_.valid(); }
};

// Create an immediately-resolved future (no thread launch overhead).
inline legion_future<void> make_ready_future() {
    std::promise<void> p;
    p.set_value();
    return legion_future<void>(p.get_future());
}

// Asynchronous launch utility.  In a full Legion deployment each of these
// could be mapped to a child task launch through the Legion runtime; here
// we use std::async to keep the STM internals self-contained.
template <typename F, typename... Args>
inline legion_future<void> async_launch(F&& f, Args&&... args) {
    return legion_future<void>(
        std::async(std::launch::async,
                   std::forward<F>(f),
                   std::forward<Args>(args)...)
    );
}

} // namespace detail
} // namespace astm

#define ASTM_FUTURE   astm::detail::legion_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC    astm::detail::async_launch
#define ASTM_MAKE_READY_FUTURE astm::detail::make_ready_future()

#define ASTM_TEST  assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// Secondary guard – ensure astm.hpp definitions are available when
// astm_config.hpp is included directly.
#if !defined(ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E)
#define ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E

#include "astm.hpp"

namespace astm
{
}

#endif
