////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// Legion runtime API (default mapper assumed by build/runtime configuration).
#include <legion.h>

// Host-side synchronization/future utilities used by ASTM internals.
#include <future>
#include <thread>
#include <mutex>
#include <functional>
#include <type_traits>
#include <utility>
#include <cassert>

namespace astm { namespace detail {

// A small future wrapper that provides .then(...) chaining semantics similar
// to what the HPX-based version expected.
template <typename T>
class astm_future
{
public:
    astm_future() = default;
    astm_future(std::future<T>&& f) : fut_(f.share()) {}
    astm_future(std::shared_future<T> f) : fut_(std::move(f)) {}

    T get() { return fut_.get(); }
    bool valid() const noexcept { return fut_.valid(); }

    template <typename F>
    auto then(F&& f) -> astm_future<std::invoke_result_t<std::decay_t<F>&>>
    {
        using R = std::invoke_result_t<std::decay_t<F>&>;
        auto next = std::async(
            std::launch::async,
            [prev = fut_, func = std::forward<F>(f)]() mutable -> R {
                if (prev.valid()) prev.wait();
                if constexpr (std::is_void_v<R>) {
                    func();
                    return;
                } else {
                    return func();
                }
            });
        return astm_future<R>(std::move(next));
    }

private:
    std::shared_future<T> fut_;
};

template <>
class astm_future<void>
{
public:
    astm_future() = default;
    astm_future(std::future<void>&& f) : fut_(f.share()) {}
    astm_future(std::shared_future<void> f) : fut_(std::move(f)) {}

    void get()
    {
        if (fut_.valid()) fut_.get();
    }

    bool valid() const noexcept { return fut_.valid(); }

    template <typename F>
    auto then(F&& f) -> astm_future<std::invoke_result_t<std::decay_t<F>&>>
    {
        using R = std::invoke_result_t<std::decay_t<F>&>;
        auto next = std::async(
            std::launch::async,
            [prev = fut_, func = std::forward<F>(f)]() mutable -> R {
                if (prev.valid()) prev.wait();
                if constexpr (std::is_void_v<R>) {
                    func();
                    return;
                } else {
                    return func();
                }
            });
        return astm_future<R>(std::move(next));
    }

private:
    std::shared_future<void> fut_;
};

template <typename F, typename... Args>
auto async(F&& f, Args&&... args)
    -> astm_future<std::invoke_result_t<F, Args...>>
{
    using R = std::invoke_result_t<F, Args...>;
    auto sf = std::async(
        std::launch::async,
        std::forward<F>(f),
        std::forward<Args>(args)...);
    return astm_future<R>(std::move(sf));
}

inline astm_future<void> make_ready_future()
{
    std::promise<void> p;
    p.set_value();
    return astm_future<void>(p.get_future());
}

}} // namespace astm::detail

// ASTM configuration macros for Legion-targeted builds.
#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<ASTM_MUTEX>
#define ASTM_FUTURE astm::detail::astm_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC astm::detail::async
#define ASTM_MAKE_READY_FUTURE astm::detail::make_ready_future

#define ASTM_TEST assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95