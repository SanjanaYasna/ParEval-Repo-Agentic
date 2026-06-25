////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// Legion runtime (default mapper assumed by the application build/run setup).
#include <legion.h>

#include <future>
#include <thread>
#include <mutex>
#include <functional>
#include <type_traits>
#include <utility>
#include <cassert>

namespace astm { namespace detail {

// A small future wrapper that provides HPX-like `.then(...)` chaining semantics.
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

    legion_future(legion_future const&) = default;
    legion_future(legion_future&&) noexcept = default;
    legion_future& operator=(legion_future const&) = default;
    legion_future& operator=(legion_future&&) noexcept = default;

    void wait() const
    {
        fut_.wait();
    }

    decltype(auto) get()
    {
        return fut_.get();
    }

    bool valid() const noexcept
    {
        return fut_.valid();
    }

    template <typename F>
    auto then(F&& f)
      -> legion_future<typename std::invoke_result_t<std::decay_t<F>>>
    {
        using result_type = typename std::invoke_result_t<std::decay_t<F>>;
        auto prev = fut_;

        auto cont = std::async(std::launch::async,
            [prev, fn = std::forward<F>(f)]() mutable -> result_type {
                prev.wait();
                if constexpr (std::is_void_v<T>) {
                    prev.get();
                } else {
                    (void)prev.get();
                }

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

inline legion_future<void> make_ready_future()
{
    std::promise<void> p;
    p.set_value();
    return legion_future<void>(p.get_future());
}

template <typename F, typename... Args>
auto async(F&& f, Args&&... args)
  -> legion_future<typename std::invoke_result_t<F, Args...>>
{
    using result_type = typename std::invoke_result_t<F, Args...>;
    return legion_future<result_type>(
        std::async(std::launch::async,
                   std::forward<F>(f),
                   std::forward<Args>(args)...));
}

}} // namespace astm::detail

// ASTM backend configuration (Legion-targeted build).
#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<ASTM_MUTEX>
#define ASTM_FUTURE astm::detail::legion_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC astm::detail::async
#define ASTM_MAKE_READY_FUTURE astm::detail::make_ready_future

#define ASTM_TEST assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95
