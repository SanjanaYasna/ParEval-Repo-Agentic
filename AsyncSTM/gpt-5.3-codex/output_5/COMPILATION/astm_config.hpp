////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// Legion runtime (default mapper assumed by the build/run environment).
#include "legion.h"

#include <cassert>
#include <functional>
#include <future>
#include <mutex>
#include <type_traits>
#include <utility>

namespace astm { namespace detail {

// A small future wrapper that provides .then(...) semantics expected by ASTM.
// This keeps the existing ASTM interface intact while targeting Legion builds.
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

    void wait() const
    {
        fut_.wait();
    }

    bool valid() const
    {
        return fut_.valid();
    }

    template <typename F>
    auto then(F&& f) const
      -> legion_future<typename std::invoke_result_t<std::decay_t<F>>>
    {
        using R = typename std::invoke_result_t<std::decay_t<F>>;
        auto prev = fut_;

        auto cont = std::async(std::launch::async,
            [prev, fn = std::forward<F>(f)]() mutable -> R
            {
                prev.wait();
                (void) prev.get(); // propagate exceptions from predecessor
                if constexpr (std::is_void<R>::value)
                {
                    std::invoke(fn);
                    return;
                }
                else
                {
                    return std::invoke(fn);
                }
            });

        return legion_future<R>(std::move(cont));
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

    void wait() const
    {
        fut_.wait();
    }

    bool valid() const
    {
        return fut_.valid();
    }

    template <typename F>
    auto then(F&& f) const
      -> legion_future<typename std::invoke_result_t<std::decay_t<F>>>
    {
        using R = typename std::invoke_result_t<std::decay_t<F>>;
        auto prev = fut_;

        auto cont = std::async(std::launch::async,
            [prev, fn = std::forward<F>(f)]() mutable -> R
            {
                prev.wait();
                prev.get(); // propagate exceptions from predecessor
                if constexpr (std::is_void<R>::value)
                {
                    std::invoke(fn);
                    return;
                }
                else
                {
                    return std::invoke(fn);
                }
            });

        return legion_future<R>(std::move(cont));
    }

private:
    std::shared_future<void> fut_;
};

template <typename F, typename... Args>
auto async(F&& f, Args&&... args)
  -> legion_future<typename std::invoke_result_t<F, Args...>>
{
    using R = typename std::invoke_result_t<F, Args...>;
    auto sf = std::async(std::launch::async,
                         std::forward<F>(f),
                         std::forward<Args>(args)...);
    return legion_future<R>(std::move(sf));
}

inline legion_future<void> make_ready_future()
{
    std::promise<void> p;
    p.set_value();
    return legion_future<void>(p.get_future());
}

template <typename T>
legion_future<typename std::decay<T>::type> make_ready_future(T&& value)
{
    using U = typename std::decay<T>::type;
    std::promise<U> p;
    p.set_value(std::forward<T>(value));
    return legion_future<U>(p.get_future());
}

}} // namespace astm::detail

// ASTM compatibility macros
#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<ASTM_MUTEX>
#define ASTM_FUTURE astm::detail::legion_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC astm::detail::async
#define ASTM_MAKE_READY_FUTURE astm::detail::make_ready_future

#define ASTM_TEST assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95
