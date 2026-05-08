////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// Legion backend (default for this translated codebase).
// NOTE:
// - We include Legion headers to align this configuration with Legion builds.
// - ASTM's current async API expects continuation-style futures (`then`), which
//   Legion's C++ Future type does not directly provide in this form.
// - To preserve ASTM semantics with minimal invasive change, we provide a
//   Legion-compatible shim future/async layer backed by std::async.
#include <legion.h>

#include <future>
#include <thread>
#include <mutex>
#include <functional>
#include <type_traits>
#include <utility>
#include <cassert>

namespace astm { namespace detail {

template <typename T>
class legion_future
{
public:
    using value_type = T;

    legion_future()
      : fut_(make_ready_shared_future())
    {}

    explicit legion_future(std::future<T>&& fut)
      : fut_(fut.share())
    {}

    explicit legion_future(std::shared_future<T> fut)
      : fut_(std::move(fut))
    {}

    T get() const
    {
        return fut_.get();
    }

    template <typename F>
    auto then(F&& f) const
        -> legion_future<std::invoke_result_t<std::decay_t<F>, T>>
    {
        using next_type = std::invoke_result_t<std::decay_t<F>, T>;

        auto prev = fut_;
        auto next_std = std::async(
            std::launch::async,
            [prev, fn = std::forward<F>(f)]() mutable -> next_type {
                if constexpr (std::is_void_v<next_type>) {
                    std::invoke(fn, prev.get());
                    return;
                } else {
                    return std::invoke(fn, prev.get());
                }
            });

        return legion_future<next_type>(std::move(next_std));
    }

private:
    static std::shared_future<T> make_ready_shared_future()
    {
        std::promise<T> p;
        p.set_value(T{});
        return p.get_future().share();
    }

    std::shared_future<T> fut_;
};

template <>
class legion_future<void>
{
public:
    using value_type = void;

    legion_future()
      : fut_(make_ready_shared_future())
    {}

    explicit legion_future(std::future<void>&& fut)
      : fut_(fut.share())
    {}

    explicit legion_future(std::shared_future<void> fut)
      : fut_(std::move(fut))
    {}

    void get() const
    {
        fut_.wait();
        fut_.get();
    }

    template <typename F>
    auto then(F&& f) const
        -> legion_future<std::invoke_result_t<std::decay_t<F>>>
    {
        using next_type = std::invoke_result_t<std::decay_t<F>>;

        auto prev = fut_;
        auto next_std = std::async(
            std::launch::async,
            [prev, fn = std::forward<F>(f)]() mutable -> next_type {
                prev.wait();
                prev.get();
                if constexpr (std::is_void_v<next_type>) {
                    std::invoke(fn);
                    return;
                } else {
                    return std::invoke(fn);
                }
            });

        return legion_future<next_type>(std::move(next_std));
    }

private:
    static std::shared_future<void> make_ready_shared_future()
    {
        std::promise<void> p;
        p.set_value();
        return p.get_future().share();
    }

    std::shared_future<void> fut_;
};

inline legion_future<void> make_ready_future()
{
    std::promise<void> p;
    p.set_value();
    return legion_future<void>(p.get_future().share());
}

template <typename T>
inline legion_future<T> make_ready_future(T value)
{
    std::promise<T> p;
    p.set_value(std::move(value));
    return legion_future<T>(p.get_future().share());
}

template <typename F, typename... Args>
auto legion_async(F&& f, Args&&... args)
    -> legion_future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
{
    using result_type = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;
    auto fut = std::async(std::launch::async, std::forward<F>(f), std::forward<Args>(args)...);
    return legion_future<result_type>(std::move(fut));
}

}} // namespace astm::detail

#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<ASTM_MUTEX>
#define ASTM_FUTURE astm::detail::legion_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC astm::detail::legion_async
#define ASTM_MAKE_READY_FUTURE astm::detail::make_ready_future

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
