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
#include <tuple>
#include <type_traits>
#include <assert.h>

namespace astm_detail {

// A future wrapper that supports .then() chaining, analogous to HPX futures.
// Uses std::shared_future internally for copyability and
// std::packaged_task + detached threads to avoid the blocking-destructor
// behavior of std::async-created futures.
template <typename T>
class chainable_future;

template <>
class chainable_future<void> {
    std::shared_future<void> impl_;
public:
    chainable_future() : impl_() {}

    chainable_future(std::future<void>&& f)
        : impl_(f.share()) {}

    chainable_future(std::shared_future<void> f)
        : impl_(std::move(f)) {}

    chainable_future(const chainable_future&) = default;
    chainable_future(chainable_future&&) = default;
    chainable_future& operator=(const chainable_future&) = default;
    chainable_future& operator=(chainable_future&&) = default;

    void get() {
        if (impl_.valid()) impl_.get();
    }

    // Chains a continuation: waits for the current future to complete,
    // then executes f, returning a new future for the combined operation.
    // Mirrors hpx::future<void>::then() semantics where the callable
    // receives no arguments (or extra arguments are ignored via std::bind).
    template <typename F>
    chainable_future<void> then(F f) {
        auto prev = impl_;
        auto task = std::make_shared<std::packaged_task<void()>>(
            [prev, func = std::move(f)]() mutable {
                if (prev.valid()) prev.get();
                func();
            }
        );
        auto fut = task->get_future();
        std::thread([task]() { (*task)(); }).detach();
        return chainable_future<void>(std::move(fut));
    }
};

// Creates an immediately-ready chainable_future<void>.
// Used as: ASTM_MAKE_READY_FUTURE() — matching the HPX convention
// where the macro expands to a callable name.
inline chainable_future<void> make_ready_future() {
    std::promise<void> p;
    p.set_value();
    return chainable_future<void>(p.get_future());
}

// Launches an asynchronous operation on a detached thread,
// returning a chainable_future for the result.
// Uses std::packaged_task (not std::async) so that discarding the
// returned future does not block — enabling fire-and-forget semantics
// equivalent to hpx::async / hpx::apply.
// A lambda with std::apply is used instead of nested std::bind to
// avoid special treatment of bind sub-expressions.
template<typename F, typename... Args>
chainable_future<void> legion_async(F&& f, Args&&... args) {
    auto callable = [fn = std::decay_t<F>(std::forward<F>(f)),
                     args_tuple = std::make_tuple(std::forward<Args>(args)...)]() mutable {
        std::apply(std::move(fn), std::move(args_tuple));
    };
    auto task = std::make_shared<std::packaged_task<void()>>(std::move(callable));
    chainable_future<void> result(task->get_future());
    std::thread([task]() { (*task)(); }).detach();
    return result;
}

} // namespace astm_detail

#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<std::mutex>
#define ASTM_FUTURE astm_detail::chainable_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC astm_detail::legion_async
#define ASTM_MAKE_READY_FUTURE astm_detail::make_ready_future

#define ASTM_TEST assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// Secondary include guard (from original) for forward-declaring
// the astm namespace after astm.hpp is available.
#if !defined(ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E)
#define ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E

#include "astm.hpp"

namespace astm
{
}

#endif
