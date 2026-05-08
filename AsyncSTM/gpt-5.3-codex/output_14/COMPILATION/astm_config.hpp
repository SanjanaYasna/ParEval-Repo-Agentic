////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// Legion runtime (default mapper assumed externally).
#include <legion.h>

#include <cassert>
#include <functional>
#include <future>
#include <mutex>
#include <type_traits>
#include <utility>

namespace astm { namespace detail {

// Minimal future wrapper with HPX-like API surface used by this codebase.
template <typename T>
class legion_future
{
public:
    legion_future() = default;
    explicit legion_future(std::future<T>&& f) : fut_(f.share()) {}
    explicit legion_future(std::shared_future<T> f) : fut_(std::move(f)) {}

    T get() { return fut_.get(); }
    void wait() const { if (fut_.valid()) fut_.wait(); }

    template <typename F>
    auto then(F&& f) -> legion_future<std::invoke_result_t<F>>
    {
        using result_type = std::invoke_result_t<F>;
        auto prev = fut_;

        auto cont = std::async(std::launch::async,
            [prev, fn = std::forward<F>(f)]() mutable -> result_type {
                if (prev.valid()) (void)prev.get();
                if constexpr (std::is_void_v<result_type>) {
                    fn();
                    return;
                } else {
                    return fn();
                }
            });

        return legion_future<result_type>(std::move(cont));
    }

private:
    std::shared_future<T> fut_;
};

template <>
class legion_future<void>
{
public:
    legion_future() = default;
    explicit legion_future(std::future<void>&& f) : fut_(f.share()) {}
    explicit legion_future(std::shared_future<void> f) : fut_(std::move(f)) {}

    void get() { if (fut_.valid()) fut_.get(); }
    void wait() const { if (fut_.valid()) fut_.wait(); }

    template <typename F>
    auto then(F&& f) -> legion_future<std::invoke_result_t<F>>
    {
        using result_type = std::invoke_result_t<F>;
        auto prev = fut_;

        auto cont = std::async(std::launch::async,
            [prev, fn = std::forward<F>(f)]() mutable -> result_type {
                if (prev.valid()) prev.get();
                if constexpr (std::is_void_v<result_type>) {
                    fn();
                    return;
                } else {
                    return fn();
                }
            });

        return legion_future<result_type>(std::move(cont));
    }

private:
    std::shared_future<void> fut_;
};

template <typename F, typename... Args>
auto legion_async(F&& f, Args&&... args)
    -> legion_future<std::invoke_result_t<F, Args...>>
{
    using result_type = std::invoke_result_t<F, Args...>;
    auto fut = std::async(
        std::launch::async,
        std::forward<F>(f),
        std::forward<Args>(args)...);
    return legion_future<result_type>(std::move(fut));
}

inline legion_future<void> make_ready_future()
{
    std::promise<void> p;
    p.set_value();
    return legion_future<void>(p.get_future());
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
