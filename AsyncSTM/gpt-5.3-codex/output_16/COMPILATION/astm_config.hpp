////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// Legion runtime (default mapper assumed by build/run configuration)
#include <legion.h>

// Host-side synchronization/future utilities used by ASTM internals.
#include <future>
#include <mutex>
#include <functional>
#include <type_traits>
#include <utility>
#include <cassert>

namespace astm { namespace detail {

template <typename T>
class legion_future;

template <>
class legion_future<void>
{
public:
    legion_future() = default;
    explicit legion_future(std::future<void>&& f) : fut_(f.share()) {}
    explicit legion_future(std::shared_future<void> f) : fut_(std::move(f)) {}

    void get() { fut_.get(); }
    void wait() const { fut_.wait(); }
    bool valid() const { return fut_.valid(); }

    template <typename F>
    auto then(F&& f) -> legion_future<std::invoke_result_t<F>>
    {
        using result_type = std::invoke_result_t<F>;
        auto prev = fut_;

        auto next = std::async(std::launch::async,
            [prev, fn = std::forward<F>(f)]() mutable -> result_type
            {
                prev.wait();
                prev.get();

                if constexpr (std::is_void_v<result_type>)
                {
                    fn();
                }
                else
                {
                    return fn();
                }
            });

        return legion_future<result_type>(std::move(next));
    }

private:
    std::shared_future<void> fut_;
};

template <typename T>
class legion_future
{
public:
    legion_future() = default;
    explicit legion_future(std::future<T>&& f) : fut_(f.share()) {}
    explicit legion_future(std::shared_future<T> f) : fut_(std::move(f)) {}

    T get() { return fut_.get(); }
    void wait() const { fut_.wait(); }
    bool valid() const { return fut_.valid(); }

    template <typename F>
    auto then(F&& f) -> legion_future<std::invoke_result_t<F>>
    {
        using result_type = std::invoke_result_t<F>;
        auto prev = fut_;

        auto next = std::async(std::launch::async,
            [prev, fn = std::forward<F>(f)]() mutable -> result_type
            {
                prev.wait();
                (void)prev.get();

                if constexpr (std::is_void_v<result_type>)
                {
                    fn();
                }
                else
                {
                    return fn();
                }
            });

        return legion_future<result_type>(std::move(next));
    }

private:
    std::shared_future<T> fut_;
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
auto astm_async(F&& f, Args&&... args)
    -> legion_future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
{
    using result_type = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;
    auto fut = std::async(std::launch::async,
                          std::forward<F>(f),
                          std::forward<Args>(args)...);
    return legion_future<result_type>(std::move(fut));
}

}} // namespace astm::detail

// ASTM configuration macros for the Legion port.
#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<ASTM_MUTEX>
#define ASTM_FUTURE astm::detail::legion_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC astm::detail::astm_async
#define ASTM_MAKE_READY_FUTURE astm::detail::make_ready_future

#define ASTM_TEST assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95
