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

namespace astm_detail {

// A future wrapper supporting .then() chaining, as required by the ASTM framework.
// Uses std::shared_future internally for copyability and safe reference counting.
template <typename T>
class chainable_future;

template <>
class chainable_future<void> {
    std::shared_future<void> fut_;
public:
    chainable_future() : fut_() {}
    chainable_future(std::future<void>&& f) : fut_(f.share()) {}
    chainable_future(std::shared_future<void>&& f) : fut_(std::move(f)) {}
    chainable_future(const std::shared_future<void>& f) : fut_(f) {}

    chainable_future(const chainable_future&) = default;
    chainable_future(chainable_future&&) = default;
    chainable_future& operator=(const chainable_future&) = default;
    chainable_future& operator=(chainable_future&&) = default;

    void get() {
        if (fut_.valid()) fut_.wait();
    }

    bool valid() const { return fut_.valid(); }

    // Chain a continuation onto this future. Returns a new future representing
    // the completion of both this future and the continuation.
    template <typename F>
    chainable_future<void> then(F&& f) {
        auto captured = fut_;
        return chainable_future<void>(
            std::async(std::launch::async,
                [captured, func = std::forward<F>(f)]() mutable {
                    if (captured.valid()) captured.wait();
                    func();
                }));
    }
};

// Launch an asynchronous operation, returning a chainable_future.
template <typename F, typename... Args>
chainable_future<void> legion_async(F&& f, Args&&... args) {
    return chainable_future<void>(
        std::async(std::launch::async,
            std::forward<F>(f), std::forward<Args>(args)...));
}

// Create an immediately-ready future.
inline chainable_future<void> make_ready_future() {
    std::promise<void> p;
    p.set_value();
    return chainable_future<void>(p.get_future());
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

//newer ASTM 
#if !defined(ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E)
#define ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E

#include "astm.hpp"

namespace astm
{
}

#endif
