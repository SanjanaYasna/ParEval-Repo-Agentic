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

    // Legion manages coherence via regions, but the STM library requires
    // fine-grained local locking for its commit protocol, so we use std::mutex.
    #define ASTM_MUTEX std::mutex
    #define ASTM_LOCK std::unique_lock<std::mutex>
    #define ASTM_FUNCTION std::function

    namespace astm { namespace detail {

    // Legion's Future<T> does not natively support .then() continuation
    // chaining. We provide a lightweight chainable future built on top of
    // std::shared_future that supports the .then() interface required by ASTM's
    // commit_transaction() and transaction_future.
    template <typename T>
    class chainable_future;

    template <>
    class chainable_future<void>
    {
        std::shared_future<void> fut_;

    public:
        chainable_future()
          : fut_()
        {}

        chainable_future(std::future<void>&& f)
          : fut_(f.share())
        {}

        chainable_future(std::shared_future<void> f)
          : fut_(std::move(f))
        {}

        chainable_future(chainable_future&&) = default;
        chainable_future(chainable_future const&) = default;
        chainable_future& operator=(chainable_future&&) = default;
        chainable_future& operator=(chainable_future const&) = default;

        void get()
        {
            if (fut_.valid())
                fut_.get();
        }

        // .then(F) waits for this future to become ready, then invokes F(),
        // and returns a new chainable_future representing completion of F.
        // This mirrors hpx::future<void>::then() as used by the ASTM library,
        // where continuations are always void()->void callables produced by
        // std::bind.
        template <typename F>
        chainable_future<void> then(F&& f)
        {
            std::shared_future<void> captured = fut_;
            std::function<void()> func(std::forward<F>(f));
            return chainable_future<void>(
                std::async(std::launch::async,
                    [captured, func]()
                    {
                        if (captured.valid())
                            captured.get();
                        func();
                    })
            );
        }
    };

    // Creates an immediately-ready chainable_future<void>.
    inline chainable_future<void> make_ready()
    {
        std::promise<void> p;
        p.set_value();
        return chainable_future<void>(p.get_future());
    }

    // Launches an asynchronous operation and returns a chainable_future<void>.
    // This replaces hpx::async in the Legion execution model.
    template <typename F, typename... Args>
    inline chainable_future<void> async_launch(F&& f, Args&&... args)
    {
        return chainable_future<void>(
            std::async(std::launch::async,
                       std::forward<F>(f),
                       std::forward<Args>(args)...)
        );
    }

    }} // namespace astm::detail

    #define ASTM_FUTURE astm::detail::chainable_future
    #define ASTM_ASYNC astm::detail::async_launch
    #define ASTM_MAKE_READY_FUTURE astm::detail::make_ready

    #define ASTM_TEST assert
    #define ASTM_REPORT 0

#elif defined(ASTM_HPX) // Use hpx::
    #include <hpx/include/lcos.hpp>
    #include <hpx/async.hpp>
    #include <hpx/synchronization/spinlock_pool.hpp>
    #define ASTM_MUTEX hpx::util::spinlock
    //std shared lock mimicry 
    #define ASTM_LOCK std::unique_lock<ASTM_MUTEX> 
    #define ASTM_FUTURE hpx::future
    #define ASTM_FUNCTION std::function
    #define ASTM_ASYNC hpx::async
    #define ASTM_MAKE_READY_FUTURE hpx::make_ready_future

    #include <hpx/util/lightweight_test.hpp>

    #define ASTM_TEST HPX_TEST
    #define ASTM_REPORT hpx::util::report_errors()
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
