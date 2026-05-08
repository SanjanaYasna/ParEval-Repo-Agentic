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

// ---------------------------------------------------------------------------
// Legion-compatible future wrapper that supports .then() chaining, used
// internally by the STM library for continuation-based async operations.
// Legion manages coarse-grained parallelism through tasks and regions;
// the STM commit protocol still needs fine-grained locking and future
// chaining which are provided here via std:: primitives.
// ---------------------------------------------------------------------------
namespace astm { namespace detail {

// Forward declaration of the general template.
template <typename T> class legion_future;

// Explicit specialization for void.
template <>
class legion_future<void>
{
    std::shared_future<void> fut_;

  public:
    legion_future() {}

    legion_future(std::future<void>&& f)
      : fut_(f.share()) {}

    legion_future(std::shared_future<void> const& f)
      : fut_(f) {}

    legion_future(std::shared_future<void>&& f)
      : fut_(std::move(f)) {}

    legion_future(legion_future const&)            = default;
    legion_future(legion_future&&)                 = default;
    legion_future& operator=(legion_future const&) = default;
    legion_future& operator=(legion_future&&)      = default;

    /// Chain a continuation that runs after this future is ready.
    /// Returns a new future representing the completion of the continuation.
    template <typename F>
    legion_future<void> then(F&& f)
    {
        auto predecessor = fut_;
        return legion_future<void>(
            std::async(std::launch::async,
                [predecessor, func = std::forward<F>(f)]() mutable {
                    if (predecessor.valid())
                        predecessor.get();
                    func();
                })
        );
    }

    void get()
    {
        if (fut_.valid())
            fut_.get();
    }

    bool valid() const { return fut_.valid(); }
};

// Primary template for non-void types.
template <typename T>
class legion_future
{
    std::shared_future<T> fut_;

  public:
    legion_future() {}

    legion_future(std::future<T>&& f)
      : fut_(f.share()) {}

    legion_future(std::shared_future<T> const& f)
      : fut_(f) {}

    legion_future(std::shared_future<T>&& f)
      : fut_(std::move(f)) {}

    legion_future(legion_future const&)            = default;
    legion_future(legion_future&&)                 = default;
    legion_future& operator=(legion_future const&) = default;
    legion_future& operator=(legion_future&&)      = default;

    template <typename F>
    legion_future<void> then(F&& f)
    {
        auto predecessor = fut_;
        return legion_future<void>(
            std::async(std::launch::async,
                [predecessor, func = std::forward<F>(f)]() mutable {
                    if (predecessor.valid())
                        predecessor.get();
                    func();
                })
        );
    }

    T get()
    {
        if (fut_.valid())
            return fut_.get();
        return T();
    }

    bool valid() const { return fut_.valid(); }
};

// Create an immediately-ready void future (analogous to hpx::make_ready_future).
inline legion_future<void> make_ready_future()
{
    std::promise<void> p;
    p.set_value();
    return legion_future<void>(p.get_future());
}

// Launch an asynchronous operation, returning a legion_future<void>.
// Uses std::async internally; Legion manages higher-level task parallelism.
template <typename F, typename... Args>
legion_future<void> legion_async(F&& f, Args&&... args)
{
    return legion_future<void>(
        std::async(std::launch::async,
                   std::forward<F>(f),
                   std::forward<Args>(args)...)
    );
}

}} // namespace astm::detail

// ---- Macro definitions for the Legion execution model ---------------------

#define ASTM_MUTEX    std::mutex
#define ASTM_LOCK     std::unique_lock<std::mutex>
#define ASTM_FUTURE   astm::detail::legion_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC    astm::detail::legion_async
// Note: defined as a function name so that ASTM_MAKE_READY_FUTURE() works
// (matches hpx::make_ready_future usage).
#define ASTM_MAKE_READY_FUTURE astm::detail::make_ready_future

#define ASTM_TEST   assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// ---------------------------------------------------------------------------
// Newer ASTM guard – kept for compatibility with the original header layout.
// ---------------------------------------------------------------------------
#if !defined(ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E)
#define ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E

#include "astm.hpp"

namespace astm
{
}

#endif
