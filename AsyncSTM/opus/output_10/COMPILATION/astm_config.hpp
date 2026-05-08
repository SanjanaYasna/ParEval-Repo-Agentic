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
#include <fstream>

// Within Legion tasks, standard C++ mutual-exclusion primitives are used
// for the STM's internal locking protocol.
#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<std::mutex>
#define ASTM_FUNCTION std::function

namespace astm {
namespace detail {

// ---------------------------------------------------------------------------
// A future wrapper that provides .then() chaining.
// Neither Legion::Future nor std::future support .then(); HPX::future did,
// so we implement a compatible wrapper on top of std::future + std::async.
// ---------------------------------------------------------------------------
template<typename T>
class legion_future;

template<>
class legion_future<void> {
    std::future<void> fut_;
public:
    legion_future() {}

    legion_future(std::future<void>&& f) : fut_(std::move(f)) {}

    legion_future(legion_future&& other) noexcept
      : fut_(std::move(other.fut_)) {}

    legion_future& operator=(legion_future&& other) noexcept {
        if (this != &other)
            fut_ = std::move(other.fut_);
        return *this;
    }

    // Non-copyable (matches std::future / hpx::future semantics)
    legion_future(const legion_future&) = delete;
    legion_future& operator=(const legion_future&) = delete;

    // Chain a continuation: waits for the current future, then invokes func().
    // Extra arguments passed to func by std::bind are silently accepted,
    // matching HPX's .then() behaviour where the predecessor future is passed
    // to the callback (and ignored when using std::bind).
    template<typename F>
    legion_future<void> then(F&& func) {
        auto prev = std::make_shared<std::future<void>>(std::move(fut_));
        return legion_future<void>(
            std::async(std::launch::async,
                [prev, f = std::forward<F>(func)]() mutable {
                    if (prev->valid())
                        prev->get();
                    f();
                }));
    }

    void get() {
        if (fut_.valid())
            fut_.get();
    }

    bool valid() const { return fut_.valid(); }
};

// Create an immediately-ready future (replaces hpx::make_ready_future).
// Defined as a callable so that ASTM_MAKE_READY_FUTURE() works the same
// way as hpx::make_ready_future().
inline legion_future<void> make_ready_future() {
    std::promise<void> p;
    p.set_value();
    return legion_future<void>(p.get_future());
}

// Async launcher that forces std::launch::async to ensure true parallel
// execution within Legion tasks (mirrors hpx::async behaviour).
template<typename F, typename... Args>
auto legion_async(F&& f, Args&&... args)
{
    return std::async(std::launch::async,
        std::forward<F>(f), std::forward<Args>(args)...);
}

} // namespace detail
} // namespace astm

#define ASTM_FUTURE astm::detail::legion_future
#define ASTM_ASYNC astm::detail::legion_async
#define ASTM_MAKE_READY_FUTURE astm::detail::make_ready_future

#define ASTM_TEST assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// Additional ASTM namespace extensions
#if !defined(ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E)
#define ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E

#include "astm.hpp"

namespace astm
{
}

#endif
