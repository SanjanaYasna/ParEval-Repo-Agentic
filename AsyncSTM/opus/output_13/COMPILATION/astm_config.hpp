////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

#include "legion.h"

#include <future>
#include <thread>
#include <mutex>
#include <functional>
#include <memory>
#include <cassert>
#include <fstream>
#include <type_traits>

namespace astm_detail {

// Forward declaration
template <typename T>
class legion_future;

// Void specialization — supports .then() chaining and .get()
template <>
class legion_future<void> {
    std::shared_future<void> fut_;
public:
    legion_future() : fut_() {}
    legion_future(std::future<void>&& f) : fut_(f.share()) {}
    legion_future(std::shared_future<void> f) : fut_(std::move(f)) {}
    legion_future(const legion_future&) = default;
    legion_future(legion_future&&) = default;
    legion_future& operator=(const legion_future&) = default;
    legion_future& operator=(legion_future&&) = default;

    void get() {
        if (fut_.valid()) fut_.get();
    }

    // Chain a continuation after this future completes.
    // F must be callable with zero arguments (matching std::bind usage in commit_transaction).
    template <typename F>
    legion_future<void> then(F f) {
        auto captured_fut = fut_;
        return legion_future<void>(
            std::async(std::launch::async,
                [captured_fut, func = std::move(f)]() mutable {
                    if (captured_fut.valid()) captured_fut.wait();
                    func();
                }
            )
        );
    }
};

// General template
template <typename T>
class legion_future {
    std::shared_future<T> fut_;
public:
    legion_future() : fut_() {}
    legion_future(std::future<T>&& f) : fut_(f.share()) {}
    legion_future(std::shared_future<T> f) : fut_(std::move(f)) {}
    legion_future(const legion_future&) = default;
    legion_future(legion_future&&) = default;
    legion_future& operator=(const legion_future&) = default;
    legion_future& operator=(legion_future&&) = default;

    T get() { return fut_.get(); }

    template <typename F>
    auto then(F f) -> legion_future<decltype(f())> {
        auto captured_fut = fut_;
        return legion_future<decltype(f())>(
            std::async(std::launch::async,
                [captured_fut, func = std::move(f)]() mutable {
                    if (captured_fut.valid()) captured_fut.wait();
                    return func();
                }
            )
        );
    }
};

// Create an immediately-ready void future.
inline legion_future<void> make_ready_legion_future() {
    std::promise<void> p;
    p.set_value();
    return legion_future<void>(std::move(p.get_future()));
}

// Launch an asynchronous task, returning a legion_future.
template <typename F, typename... Args>
auto legion_async(F&& f, Args&&... args)
    -> legion_future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
{
    using result_type = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;
    return legion_future<result_type>(
        std::async(std::launch::async, std::forward<F>(f), std::forward<Args>(args)...)
    );
}

} // namespace astm_detail

// Mutex / lock — used for intra-task synchronization during transaction commit
#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<std::mutex>

// Future / async — wrapper providing .then() chaining on top of std::shared_future
#define ASTM_FUTURE astm_detail::legion_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC astm_detail::legion_async
#define ASTM_MAKE_READY_FUTURE astm_detail::make_ready_legion_future

// Testing helpers
#define ASTM_TEST assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// Newer ASTM section
#if !defined(ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E)
#define ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E

#include "astm.hpp"

namespace astm
{
}

#endif
