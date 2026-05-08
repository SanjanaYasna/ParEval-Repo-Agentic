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
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>

namespace astm { namespace legion_backend {

// A small future wrapper that provides HPX-like .then(...) chaining semantics
// used by astm.hpp, while remaining C++17-compatible.
template <typename T>
class future
{
public:
    future() = default;
    explicit future(std::future<T>&& f) : fut_(f.share()) {}
    explicit future(std::shared_future<T> f) : fut_(std::move(f)) {}

    T get() { return fut_.get(); }
    void wait() const
    {
        if (fut_.valid()) fut_.wait();
    }
    bool valid() const { return fut_.valid(); }

    template <typename F>
    future<void> then(F&& f);

private:
    std::shared_future<T> fut_;
};

template <>
class future<void>
{
public:
    future() = default;
    explicit future(std::future<void>&& f) : fut_(f.share()) {}
    explicit future(std::shared_future<void> f) : fut_(std::move(f)) {}

    void get()
    {
        if (fut_.valid()) fut_.get();
    }
    void wait() const
    {
        if (fut_.valid()) fut_.wait();
    }
    bool valid() const { return fut_.valid(); }

    template <typename F>
    future<void> then(F&& f)
    {
        auto prev = fut_;
        return future<void>(std::async(std::launch::async,
            [prev, cont = std::forward<F>(f)]() mutable {
                if (prev.valid()) prev.wait();
                std::invoke(std::move(cont));
            }));
    }

private:
    std::shared_future<void> fut_;
};

template <typename T>
template <typename F>
future<void> future<T>::then(F&& f)
{
    auto prev = fut_;
    return future<void>(std::async(std::launch::async,
        [prev, cont = std::forward<F>(f)]() mutable {
            if (prev.valid()) prev.wait();
            std::invoke(std::move(cont));
        }));
}

inline future<void> make_ready_future()
{
    std::promise<void> p;
    p.set_value();
    return future<void>(p.get_future());
}

template <typename F, typename... Args>
auto async(F&& f, Args&&... args)
    -> future<std::invoke_result_t<F, Args...>>
{
    using result_type = std::invoke_result_t<F, Args...>;
    return future<result_type>(
        std::async(std::launch::async,
                   std::forward<F>(f),
                   std::forward<Args>(args)...));
}

}} // namespace astm::legion_backend

// ASTM compatibility macros used across the existing codebase.
#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<ASTM_MUTEX>
#define ASTM_FUTURE astm::legion_backend::future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC astm::legion_backend::async
#define ASTM_MAKE_READY_FUTURE astm::legion_backend::make_ready_future

#define ASTM_TEST assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95
