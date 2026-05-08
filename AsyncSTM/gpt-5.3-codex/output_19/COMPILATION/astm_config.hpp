////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

#include <legion.h>

#include <cassert>
#include <functional>
#include <future>
#include <mutex>
#include <type_traits>
#include <utility>

namespace astm
{

// Lightweight future wrapper used by ASTM to preserve HPX-like .then() chaining.
template <typename T>
class legion_future
{
public:
    legion_future() = default;

    explicit legion_future(std::future<T>&& f)
      : state_(f.share())
    {}

    explicit legion_future(std::shared_future<T> f)
      : state_(std::move(f))
    {}

    T get()
    {
        return state_.get();
    }

    void wait() const
    {
        state_.wait();
    }

    template <typename F>
    auto then(F&& f) const
      -> legion_future<typename std::invoke_result_t<std::decay_t<F>>>
    {
        using R = typename std::invoke_result_t<std::decay_t<F>>;

        auto prev = state_;
        auto next = std::async(
            std::launch::async,
            [prev, cont = std::decay_t<F>(std::forward<F>(f))]() mutable -> R {
                prev.wait();
                if constexpr (std::is_void_v<R>)
                {
                    cont();
                    return;
                }
                else
                {
                    return cont();
                }
            });

        return legion_future<R>(std::move(next));
    }

private:
    std::shared_future<T> state_;
};

template <>
class legion_future<void>
{
public:
    legion_future() = default;

    explicit legion_future(std::future<void>&& f)
      : state_(f.share())
    {}

    explicit legion_future(std::shared_future<void> f)
      : state_(std::move(f))
    {}

    void get()
    {
        state_.get();
    }

    void wait() const
    {
        state_.wait();
    }

    template <typename F>
    auto then(F&& f) const
      -> legion_future<typename std::invoke_result_t<std::decay_t<F>>>
    {
        using R = typename std::invoke_result_t<std::decay_t<F>>;

        auto prev = state_;
        auto next = std::async(
            std::launch::async,
            [prev, cont = std::decay_t<F>(std::forward<F>(f))]() mutable -> R {
                prev.wait();
                if constexpr (std::is_void_v<R>)
                {
                    cont();
                    return;
                }
                else
                {
                    return cont();
                }
            });

        return legion_future<R>(std::move(next));
    }

private:
    std::shared_future<void> state_;
};

template <typename F, typename... Args>
auto legion_async(F&& f, Args&&... args)
  -> legion_future<typename std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
{
    using R = typename std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;
    return legion_future<R>(
        std::async(std::launch::async, std::forward<F>(f), std::forward<Args>(args)...));
}

inline legion_future<void> make_ready_future()
{
    std::promise<void> p;
    p.set_value();
    return legion_future<void>(p.get_future());
}

} // namespace astm

#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<ASTM_MUTEX>
#define ASTM_FUTURE astm::legion_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC astm::legion_async
#define ASTM_MAKE_READY_FUTURE astm::make_ready_future

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
