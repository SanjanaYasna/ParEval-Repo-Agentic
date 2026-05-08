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
    #include <memory>
    #include <assert.h>

    namespace astm {
    namespace detail {

    ////////////////////////////////////////////////////////////////////////////
    // A continuation-capable future wrapper.
    //
    // Legion::Future does not support .then() continuations. This wrapper
    // uses std::shared_future internally and std::async to implement the
    // continuation chaining required by the ASTM transaction commit protocol.
    // Using shared_future (rather than unique future) avoids the blocking
    // destructor behavior of std::future returned by std::async, which
    // enables proper fire-and-forget semantics in commit_transaction().
    ////////////////////////////////////////////////////////////////////////////
    template <typename T>
    class legion_future;

    template <>
    class legion_future<void>
    {
        std::shared_future<void> fut_;

    public:
        legion_future() : fut_() {}

        legion_future(std::future<void>&& f) : fut_(f.share()) {}

        legion_future(legion_future const&) = default;
        legion_future(legion_future&&) = default;
        legion_future& operator=(legion_future const&) = default;
        legion_future& operator=(legion_future&&) = default;

        void get()
        {
            if (fut_.valid()) fut_.get();
        }

        // Chain a continuation: wait for this future to complete, then
        // execute f, returning a new future for the combined work.
        template <typename F>
        legion_future then(F f)
        {
            std::shared_future<void> captured = fut_;
            return legion_future(
                std::async(std::launch::async,
                    [captured, f]() mutable {
                        if (captured.valid()) captured.get();
                        f();
                    }));
        }
    };

    // Create an immediately-ready future.
    inline legion_future<void> make_ready_future()
    {
        std::promise<void> p;
        p.set_value();
        return legion_future<void>(p.get_future());
    }

    // Launch an asynchronous operation, returning a legion_future.
    template <typename F, typename... Args>
    legion_future<void> async_launch(F&& f, Args&&... args)
    {
        return legion_future<void>(
            std::async(std::launch::async,
                       std::forward<F>(f),
                       std::forward<Args>(args)...));
    }

    } // namespace detail
    } // namespace astm

    // Legion does not provide in-task mutual-exclusion primitives;
    // std::mutex is used for STM-internal locking within a node.
    #define ASTM_MUTEX std::mutex
    // std shared lock mimicry
    #define ASTM_LOCK std::unique_lock<std::mutex>

    // Continuation-capable future (wraps std::shared_future + std::async)
    #define ASTM_FUTURE astm::detail::legion_future

    #define ASTM_FUNCTION std::function

    // Asynchronous launch returning ASTM_FUTURE<void>
    #define ASTM_ASYNC astm::detail::async_launch

    // Factory for an immediately-ready future (invoked as ASTM_MAKE_READY_FUTURE())
    #define ASTM_MAKE_READY_FUTURE astm::detail::make_ready_future

    #define ASTM_TEST assert
    #define ASTM_REPORT 0

#else                 // Use std::
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
