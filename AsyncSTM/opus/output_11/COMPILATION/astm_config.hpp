////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// Legion runtime header
#include "legion.h"

#include <future>
#include <thread>
#include <mutex>
#include <functional>
#include <memory>
#include <cassert>
#include <type_traits>

// ---------------------------------------------------------------------------
// astm_future<T>: A future wrapper providing .then() continuation chaining
// for use within the Legion execution model.
//
// Internally backed by std::shared_future so that copies needed by the
// ASTM commit protocol (capturing the antecedent in a .then() lambda while
// simultaneously move-assigning the result) work correctly.
// ---------------------------------------------------------------------------

template <typename T>
class astm_future;

// --- void specialization ---------------------------------------------------
template <>
class astm_future<void>
{
public:
    // Default-constructs an immediately-ready future.
    astm_future()
    {
        std::promise<void> p;
        p.set_value();
        fut_ = p.get_future().share();
    }

    // Construct from a std::future<void> (takes ownership).
    astm_future(std::future<void>&& f)
      : fut_(f.share())
    {}

    // Move
    astm_future(astm_future&&) noexcept            = default;
    astm_future& operator=(astm_future&&) noexcept = default;

    // Copy (shared_future is copyable)
    astm_future(astm_future const&)            = default;
    astm_future& operator=(astm_future const&) = default;

    void get()
    {
        if (fut_.valid())
            fut_.get();
    }

    // Chain a continuation.  The callable f is invoked with no arguments
    // once *this is ready.  Returns a new astm_future<void> that becomes
    // ready when f completes.
    //
    // This mirrors the HPX future::then() semantics used by the ASTM
    // commit protocol, where the continuation is a fully-bound callable
    // (std::bind absorbs the antecedent-future argument HPX would pass).
    template <typename F>
    astm_future<void> then(F f)
    {
        auto prev = fut_;                       // shared_future copy
        return astm_future<void>(
            std::async(std::launch::async,
                [prev, func = std::move(f)]() mutable {
                    if (prev.valid()) prev.get();   // wait for antecedent
                    func();                         // run continuation
                }));
    }

private:
    std::shared_future<void> fut_;
};

// --- general template ------------------------------------------------------
template <typename T>
class astm_future
{
public:
    astm_future() : fut_() {}

    astm_future(std::future<T>&& f)
      : fut_(f.share())
    {}

    astm_future(astm_future&&) noexcept            = default;
    astm_future& operator=(astm_future&&) noexcept = default;

    astm_future(astm_future const&)            = default;
    astm_future& operator=(astm_future const&) = default;

    T get()
    {
        return fut_.get();
    }

    template <typename F>
    astm_future<void> then(F f)
    {
        auto prev = fut_;
        return astm_future<void>(
            std::async(std::launch::async,
                [prev, func = std::move(f)]() mutable {
                    if (prev.valid()) prev.get();
                    func();
                }));
    }

private:
    std::shared_future<T> fut_;
};

// ---------------------------------------------------------------------------
// Async launcher – wraps std::async and returns an astm_future.
// ---------------------------------------------------------------------------
template <typename F, typename... Args>
auto astm_async_impl(F&& f, Args&&... args)
{
    using result_type =
        std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;
    return astm_future<result_type>(
        std::async(std::launch::async,
                   std::forward<F>(f),
                   std::forward<Args>(args)...));
}

// ---------------------------------------------------------------------------
// Ready-future factory (returns an immediately-ready astm_future<void>).
// ---------------------------------------------------------------------------
inline astm_future<void> astm_make_ready_future_impl()
{
    return astm_future<void>();          // default ctor is already ready
}

// ---------------------------------------------------------------------------
// ASTM macro definitions – Legion execution model
// ---------------------------------------------------------------------------

// Mutual exclusion for the STM commit protocol (intra-task)
#define ASTM_MUTEX              std::mutex
#define ASTM_LOCK               std::unique_lock<std::mutex>

// Future / async primitives
#define ASTM_FUTURE             astm_future
#define ASTM_FUNCTION           std::function
#define ASTM_ASYNC              astm_async_impl
#define ASTM_MAKE_READY_FUTURE  astm_make_ready_future_impl

// Test / reporting helpers
#define ASTM_TEST               assert
#define ASTM_REPORT             0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// ---------------------------------------------------------------------------
// Secondary include guard (preserves original file structure)
// ---------------------------------------------------------------------------
#if !defined(ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E)
#define ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E

#include "astm.hpp"

namespace astm
{
}

#endif
