////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

#if defined(ASTM_LEGION) // Use Legion execution model
    #include "legion.h"

    #include <future>
    #include <thread>
    #include <mutex>
    #include <functional>
    #include <cassert>

    // For internal STM synchronization we use standard C++ primitives
    // within Legion tasks.
    #define ASTM_MUTEX std::mutex
    #define ASTM_LOCK std::unique_lock<std::mutex>

    // ---------------------------------------------------------------------------
    // Chainable future wrapper.
    // Legion::Future does not support .then() chaining, which the ASTM
    // transaction layer relies on.  We therefore provide a thin wrapper around
    // std::shared_future that adds a .then() method backed by std::async.
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

        chainable_future(const chainable_future&)            = default;
        chainable_future(chainable_future&&)                 = default;
        chainable_future& operator=(const chainable_future&) = default;
        chainable_future& operator=(chainable_future&&)      = default;

        void get() {
            if (fut_.valid())
                fut_.get();
        }

        bool valid() const { return fut_.valid(); }

        // .then() mirrors HPX semantics: wait for the predecessor, then invoke
        // the continuation.  std::bind results silently ignore extra arguments,
        // so we simply call func() with no arguments (the bound transaction
        // pointer is already captured by std::bind).
        template <typename F>
        chainable_future<void> then(F&& f) {
            auto captured_fut = fut_;
            return chainable_future<void>(
                std::async(std::launch::async,
                    [captured_fut,
                     func = std::forward<F>(f)]() mutable {
                        if (captured_fut.valid())
                            captured_fut.wait();
                        func();
                    })
            );
        }
    };

    // Create an already-satisfied void future.
    inline chainable_future<void> make_ready() {
        std::promise<void> p;
        p.set_value();
        return chainable_future<void>(p.get_future());
    }

    // Async launcher that returns a chainable_future<void>.
    template <typename F, typename... Args>
    chainable_future<void> async_wrapper(F&& f, Args&&... args) {
        return chainable_future<void>(
            std::async(std::launch::async,
                std::forward<F>(f), std::forward<Args>(args)...)
        );
    }

    } // namespace astm_detail

    #define ASTM_FUTURE   astm_detail::chainable_future
    #define ASTM_FUNCTION std::function
    #define ASTM_ASYNC    astm_detail::async_wrapper

    // Defined as a callable name so that ASTM_MAKE_READY_FUTURE() expands to
    // astm_detail::make_ready(), matching the hpx::make_ready_future pattern.
    #define ASTM_MAKE_READY_FUTURE astm_detail::make_ready

    #define ASTM_TEST   assert
    #define ASTM_REPORT 0

#else                 // Use std:: fallback
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
