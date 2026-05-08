////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

#include <cassert>
#include <future>
#include <functional>
#include <mutex>
#include <type_traits>
#include <utility>

namespace astm { namespace legion_compat {

// A small compatibility future with `.then(...)` support.
// Backed by std::shared_future for copyability and continuation chaining.
template <typename T>
class future
{
public:
    using value_type = T;

    future() = default;

    explicit future(std::future<T>&& f)
      : shared_(f.share())
    {}

    explicit future(std::shared_future<T> f)
      : shared_(std::move(f))
    {}

    bool valid() const noexcept
    {
        return shared_.valid();
    }

    void wait() const
    {
        shared_.wait();
    }

    auto get() const
    {
        if constexpr (std::is_void_v<T>)
        {
            shared_.get();
        }
        else
        {
            return shared_.get();
        }
    }

    template <typename F>
    auto then(F&& f) const -> future<std::invoke_result_t<std::decay_t<F>>>
    {
        using fn_type = std::decay_t<F>;
        using result_type = std::invoke_result_t<fn_type>;

        auto prev = shared_;
        auto cont = std::async(
            std::launch::async,
            [prev, fn = fn_type(std::forward<F>(f))]() mutable -> result_type {
                prev.get();

                if constexpr (std::is_void_v<result_type>)
                {
                    std::invoke(fn);
                }
                else
                {
                    return std::invoke(fn);
                }
            });

        return future<result_type>(std::move(cont));
    }

private:
    std::shared_future<T> shared_;
};

inline future<void> make_ready_future()
{
    std::promise<void> p;
    p.set_value();
    return future<void>(p.get_future());
}

template <typename T>
future<T> make_ready_future(T value)
{
    std::promise<T> p;
    p.set_value(std::move(value));
    return future<T>(p.get_future());
}

template <typename F, typename... Args>
auto async(F&& f, Args&&... args)
    -> future<std::invoke_result_t<F, Args...>>
{
    using result_type = std::invoke_result_t<F, Args...>;
    auto f_std = std::async(
        std::launch::async,
        std::forward<F>(f),
        std::forward<Args>(args)...);

    return future<result_type>(std::move(f_std));
}

}} // namespace astm::legion_compat

// Runtime/synchronization config used by astm.hpp
#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<ASTM_MUTEX>
#define ASTM_FUTURE astm::legion_compat::future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC astm::legion_compat::async
#define ASTM_MAKE_READY_FUTURE astm::legion_compat::make_ready_future

#define ASTM_TEST assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95
