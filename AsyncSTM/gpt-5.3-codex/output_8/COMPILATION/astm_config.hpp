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
#include <future>
#include <mutex>
#include <functional>
#include <type_traits>
#include <utility>

namespace astm { namespace detail {

template <typename T>
class legion_future
{
public:
    legion_future() = default;
    explicit legion_future(std::future<T>&& f) : fut_(f.share()) {}
    explicit legion_future(std::shared_future<T> f) : fut_(std::move(f)) {}

    T get() { return fut_.get(); }

    template <typename F>
    auto then(F&& f) const
        -> legion_future<typename std::invoke_result_t<F, const T&>>
    {
        using result_type = typename std::invoke_result_t<F, const T&>;
        auto prev = fut_;

        auto next = std::async(std::launch::async,
            [prev, fn = std::forward<F>(f)]() mutable -> result_type {
                return fn(prev.get());
            });

        return legion_future<result_type>(std::move(next));
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

    void get() { fut_.get(); }

    template <typename F>
    legion_future<void> then(F&& f) const
    {
        auto prev = fut_;
        auto next = std::async(std::launch::async,
            [prev, fn = std::forward<F>(f)]() mutable {
                prev.get();
                fn();
            });

        return legion_future<void>(std::move(next));
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
    -> legion_future<typename std::invoke_result_t<F, Args...>>
{
    using result_type = typename std::invoke_result_t<F, Args...>;
    return legion_future<result_type>(
        std::async(std::launch::async,
                   std::forward<F>(f),
                   std::forward<Args>(args)...));
}

}} // namespace astm::detail

#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<ASTM_MUTEX>
#define ASTM_FUTURE ::astm::detail::legion_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC ::astm::detail::legion_async
#define ASTM_MAKE_READY_FUTURE ::astm::detail::make_ready_future

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
