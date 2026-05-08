////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// Legion runtime header (default mapper assumed by build/runtime setup).
#include <legion.h>

#include <cassert>
#include <functional>
#include <future>
#include <mutex>
#include <type_traits>
#include <utility>

namespace astm
{

template <typename T>
class legion_future;

template <typename F, typename... Args>
auto legion_async(F&& f, Args&&... args)
    -> legion_future<std::invoke_result_t<F, Args...>>;

// ----------------------------------------------------------------------------
// legion_future<T>
// ----------------------------------------------------------------------------
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
    auto then(F&& f) -> legion_future<std::invoke_result_t<F>>
    {
        using R = std::invoke_result_t<F>;
        using Fn = std::decay_t<F>;

        auto prev = fut_;
        return legion_async(
            [prev, fn = Fn(std::forward<F>(f))]() mutable -> R
            {
                prev.get();
                if constexpr (std::is_void_v<R>)
                {
                    fn();
                }
                else
                {
                    return fn();
                }
            });
    }

private:
    std::shared_future<T> fut_;
};

// ----------------------------------------------------------------------------
// legion_future<void> specialization
// ----------------------------------------------------------------------------
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
    auto then(F&& f) -> legion_future<std::invoke_result_t<F>>
    {
        using R = std::invoke_result_t<F>;
        using Fn = std::decay_t<F>;

        auto prev = fut_;
        return legion_async(
            [prev, fn = Fn(std::forward<F>(f))]() mutable -> R
            {
                prev.get();
                if constexpr (std::is_void_v<R>)
                {
                    fn();
                }
                else
                {
                    return fn();
                }
            });
    }

private:
    std::shared_future<void> fut_;
};

// ----------------------------------------------------------------------------
// Async helpers (host-side async inside Legion applications)
// ----------------------------------------------------------------------------
template <typename F, typename... Args>
auto legion_async(F&& f, Args&&... args)
    -> legion_future<std::invoke_result_t<F, Args...>>
{
    using R = std::invoke_result_t<F, Args...>;
    return legion_future<R>(
        std::async(std::launch::async, std::forward<F>(f), std::forward<Args>(args)...));
}

inline legion_future<void> make_ready_future()
{
    std::promise<void> p;
    p.set_value();
    return legion_future<void>(p.get_future());
}

template <typename T>
inline legion_future<std::decay_t<T>> make_ready_future(T&& value)
{
    using R = std::decay_t<T>;
    std::promise<R> p;
    p.set_value(std::forward<T>(value));
    return legion_future<R>(p.get_future());
}

} // namespace astm

// ----------------------------------------------------------------------------
// ASTM config macros for Legion translation
// ----------------------------------------------------------------------------
#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<ASTM_MUTEX>
#define ASTM_FUTURE astm::legion_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC astm::legion_async
#define ASTM_MAKE_READY_FUTURE astm::make_ready_future

#define ASTM_TEST assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95
