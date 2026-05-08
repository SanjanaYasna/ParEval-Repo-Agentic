////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// Legion headers (backend target)
#include <legion.h>

// Host-side utilities used by ASTM internals
#include <future>
#include <mutex>
#include <functional>
#include <type_traits>
#include <utility>
#include <cassert>

namespace astm
{
    // A small future wrapper with .get() and .then(...) to preserve ASTM usage.
    // This is Legion-targeted configuration, but keeps the original ASTM API shape.
    template <typename T>
    class legion_future
    {
    public:
        legion_future() = default;

        explicit legion_future(std::future<T>&& f)
          : fut_(f.share())
        {}

        explicit legion_future(std::shared_future<T> f)
          : fut_(std::move(f))
        {}

        T get() const
        {
            return fut_.get();
        }

        template <typename F>
        auto then(F&& f) const -> legion_future<std::invoke_result_t<F, T>>
        {
            using R = std::invoke_result_t<F, T>;
            auto prev = fut_;

            auto next = std::async(std::launch::async,
                [prev, fn = std::forward<F>(f)]() mutable -> R {
                    if constexpr (std::is_void_v<R>) {
                        fn(prev.get());
                    } else {
                        return fn(prev.get());
                    }
                });

            return legion_future<R>(std::move(next));
        }

    private:
        std::shared_future<T> fut_;
    };

    template <>
    class legion_future<void>
    {
    public:
        legion_future() = default;

        explicit legion_future(std::future<void>&& f)
          : fut_(f.share())
        {}

        explicit legion_future(std::shared_future<void> f)
          : fut_(std::move(f))
        {}

        void get() const
        {
            fut_.get();
        }

        template <typename F>
        auto then(F&& f) const -> legion_future<std::invoke_result_t<F>>
        {
            using R = std::invoke_result_t<F>;
            auto prev = fut_;

            auto next = std::async(std::launch::async,
                [prev, fn = std::forward<F>(f)]() mutable -> R {
                    prev.get();
                    if constexpr (std::is_void_v<R>) {
                        fn();
                    } else {
                        return fn();
                    }
                });

            return legion_future<R>(std::move(next));
        }

    private:
        std::shared_future<void> fut_;
    };

    inline legion_future<void> make_ready_future()
    {
        std::promise<void> p;
        p.set_value();
        return legion_future<void>(p.get_future());
    }

    template <typename F, typename... Args>
    auto legion_async(F&& f, Args&&... args)
      -> legion_future<std::invoke_result_t<F, Args...>>
    {
        using R = std::invoke_result_t<F, Args...>;
        auto fut = std::async(std::launch::async,
            std::forward<F>(f), std::forward<Args>(args)...);
        return legion_future<R>(std::move(fut));
    }
} // namespace astm

// ASTM backend configuration (Legion-targeted)
#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<ASTM_MUTEX>
#define ASTM_FUTURE ::astm::legion_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC ::astm::legion_async
#define ASTM_MAKE_READY_FUTURE ::astm::make_ready_future

#define ASTM_TEST assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// newer ASTM
#if !defined(ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E)
#define ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E

#include "astm.hpp"

namespace astm
{
}

#endif
