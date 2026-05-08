////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// Legion runtime headers (default mapper assumed).
#include <legion.h>

#include <future>
#include <mutex>
#include <functional>
#include <type_traits>
#include <utility>
#include <cassert>

namespace astm
{
    // A small future wrapper that provides HPX-like .then(...) chaining.
    template <typename T>
    class legion_future
    {
    public:
        legion_future() = default;

        explicit legion_future(std::future<T>&& f)
          : f_(f.share())
        {}

        explicit legion_future(std::shared_future<T> f)
          : f_(std::move(f))
        {}

        bool valid() const noexcept
        {
            return f_.valid();
        }

        decltype(auto) get()
        {
            if constexpr (std::is_void_v<T>)
            {
                f_.get();
            }
            else
            {
                return f_.get();
            }
        }

        template <typename F>
        auto then(F&& fn) -> legion_future<std::invoke_result_t<F>>
        {
            using R = std::invoke_result_t<F>;
            auto prev = f_;

            auto cont = std::async(std::launch::async,
                [prev, func = std::forward<F>(fn)]() mutable -> R
                {
                    prev.wait();
                    if constexpr (std::is_void_v<T>)
                        prev.get();
                    else
                        (void)prev.get();

                    if constexpr (std::is_void_v<R>)
                    {
                        func();
                        return;
                    }
                    else
                    {
                        return func();
                    }
                });

            return legion_future<R>(std::move(cont));
        }

    private:
        std::shared_future<T> f_;
    };

    inline legion_future<void> make_ready_future()
    {
        std::promise<void> p;
        p.set_value();
        return legion_future<void>(p.get_future().share());
    }

    template <typename T>
    legion_future<std::decay_t<T>> make_ready_future(T&& value)
    {
        using U = std::decay_t<T>;
        std::promise<U> p;
        p.set_value(std::forward<T>(value));
        return legion_future<U>(p.get_future().share());
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

// ASTM backend configuration for Legion port.
#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<ASTM_MUTEX>
#define ASTM_FUTURE astm::legion_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC astm::legion_async
#define ASTM_MAKE_READY_FUTURE astm::make_ready_future

#define ASTM_TEST assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95
