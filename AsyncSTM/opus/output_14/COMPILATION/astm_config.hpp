////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

#if defined(ASTM_LEGION)
    #include "legion.h"
    #include <future>
    #include <thread>
    #include <mutex>
    #include <functional>
    #include <memory>
    #include <cassert>

    // Legion does not replace in-task mutual exclusion primitives;
    // std::mutex is used for the STM's internal optimistic locking.
    #define ASTM_MUTEX std::mutex
    #define ASTM_LOCK std::unique_lock<std::mutex>
    #define ASTM_FUNCTION std::function

    //------------------------------------------------------------------------
    // legion_chainable_future<T>
    //
    // Neither Legion::Future nor std::future support the .then() continuation
    // chaining that ASTM requires.  This thin wrapper over std::shared_future
    // provides .then() by spawning a std::async continuation that waits on
    // the antecedent and then invokes the callable.
    //------------------------------------------------------------------------

    // Forward-declare the primary template so the void specialisation can
    // be defined first (the general template's then() returns the void
    // instantiation, so it must be a complete type at instantiation time).
    template <typename T>
    struct legion_chainable_future;

    // ---- void specialisation (the only one the ASTM library uses) ---------
    template <>
    struct legion_chainable_future<void>
    {
    private:
        std::shared_future<void> impl_;

    public:
        legion_chainable_future() : impl_() {}

        legion_chainable_future(std::future<void>&& f)
          : impl_(f.share()) {}

        legion_chainable_future(std::shared_future<void> f)
          : impl_(std::move(f)) {}

        legion_chainable_future(legion_chainable_future&&)                 = default;
        legion_chainable_future(const legion_chainable_future&)            = default;
        legion_chainable_future& operator=(legion_chainable_future&&)      = default;
        legion_chainable_future& operator=(const legion_chainable_future&) = default;

        /// Chain a continuation.  The callable \a f is invoked (with no
        /// arguments) after the antecedent future becomes ready.
        /// std::bind results silently ignore extra arguments, so this is
        /// compatible with the existing ASTM call sites.
        template <typename F>
        legion_chainable_future<void> then(F&& f)
        {
            auto captured = impl_;
            return legion_chainable_future<void>(
                std::async(std::launch::async,
                    [captured,
                     func = typename std::decay<F>::type(
                                std::forward<F>(f))]() mutable {
                        if (captured.valid()) captured.get();
                        func();
                    }));
        }

        void get()
        {
            if (impl_.valid()) impl_.get();
        }

        bool valid() const { return impl_.valid(); }
    };

    // ---- general template -------------------------------------------------
    template <typename T>
    struct legion_chainable_future
    {
    private:
        std::shared_future<T> impl_;

    public:
        legion_chainable_future() : impl_() {}

        legion_chainable_future(std::future<T>&& f)
          : impl_(f.share()) {}

        legion_chainable_future(std::shared_future<T> f)
          : impl_(std::move(f)) {}

        legion_chainable_future(legion_chainable_future&&)                 = default;
        legion_chainable_future(const legion_chainable_future&)            = default;
        legion_chainable_future& operator=(legion_chainable_future&&)      = default;
        legion_chainable_future& operator=(const legion_chainable_future&) = default;

        template <typename F>
        legion_chainable_future<void> then(F&& f)
        {
            auto captured = impl_;
            return legion_chainable_future<void>(
                std::async(std::launch::async,
                    [captured,
                     func = typename std::decay<F>::type(
                                std::forward<F>(f))]() mutable {
                        if (captured.valid()) captured.get();
                        func();
                    }));
        }

        T get()
        {
            return impl_.get();
        }

        bool valid() const { return impl_.valid(); }
    };

    #define ASTM_FUTURE legion_chainable_future

    /// Create an already-satisfied void future (replaces hpx::make_ready_future).
    inline legion_chainable_future<void> make_ready_legion_future()
    {
        std::promise<void> p;
        p.set_value();
        return legion_chainable_future<void>(p.get_future());
    }

    #define ASTM_MAKE_READY_FUTURE make_ready_legion_future

    /// Launch an asynchronous operation and return a chainable future
    /// (replaces hpx::async).
    template <typename F, typename... Args>
    inline legion_chainable_future<void> legion_async_wrapper(F&& f,
                                                              Args&&... args)
    {
        return legion_chainable_future<void>(
            std::async(std::launch::async,
                       std::forward<F>(f),
                       std::forward<Args>(args)...));
    }

    #define ASTM_ASYNC legion_async_wrapper

    #define ASTM_TEST assert
    #define ASTM_REPORT 0

#else                 // Fallback: Use std::
    #include <future>
    #include <thread>
    #include <mutex>
    #include <functional>

    #define ASTM_MUTEX std::mutex
    #define ASTM_LOCK std::unique_lock<std::mutex>
    #define ASTM_FUTURE std::future
    #define ASTM_FUNCTION std::function
    #define ASTM_ASYNC std::async

    // Note: this is a hack, no make_ready_future in std::
    #define ASTM_MAKE_READY_FUTURE std::async([](){})

    #include <assert.h>

    #define ASTM_TEST assert
    #define ASTM_REPORT 0
#endif

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

//newer ASTM 
#if !defined(ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E)
#define ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E

#include "astm.hpp"

namespace astm
{
}

#endif
