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

// A future wrapper that supports .then() chaining.
// std::future does not support .then(), but ASTM's commit_transaction()
// requires it for sequencing async operations.  We wrap std::shared_future
// so that copies are safe and .then() spawns a continuation via std::async.
template <typename T>
class chainable_future;

template <>
class chainable_future<void> {
    std::shared_future<void> fut_;

public:
    chainable_future() {}

    chainable_future(std::future<void>&& f)
      : fut_(f.share())
    {}

    chainable_future(std::shared_future<void> f)
      : fut_(std::move(f))
    {}

    chainable_future(const chainable_future&) = default;
    chainable_future(chainable_future&&) = default;
    chainable_future& operator=(const chainable_future&) = default;
    chainable_future& operator=(chainable_future&&) = default;

    void get() {
        if (fut_.valid())
            fut_.get();
    }

    bool valid() const {
        return fut_.valid();
    }

    // Chain a continuation: wait for the current future to complete,
    // then invoke f().  Returns a new chainable_future representing
    // the completion of f.
    template <typename F>
    chainable_future<void> then(F&& f) {
        auto captured = fut_;
        return chainable_future<void>(
            std::async(std::launch::async,
                [captured, func = std::forward<F>(f)]() mutable {
                    if (captured.valid())
                        captured.get();
                    func();
                })
        );
    }
};

inline chainable_future<void> make_ready_future() {
    std::promise<void> p;
    p.set_value();
    return chainable_future<void>(p.get_future());
}

// Variadic async wrapper that returns a chainable_future<void>.
// Within Legion tasks, this launches OS-level threads via std::async;
// higher-level Legion task parallelism is managed at the task-launch
// level in the translated application code.
template <typename F, typename... Args>
chainable_future<void> async_wrapper(F&& f, Args&&... args) {
    return chainable_future<void>(
        std::async(std::launch::async,
                   std::forward<F>(f),
                   std::forward<Args>(args)...)
    );
}

} // namespace astm_detail

// STM-internal synchronisation uses std:: primitives (shared-memory,
// single address-space operations within / across Legion tasks on a node).
#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<std::mutex>

// Chainable future that provides the .then() method required by
// transaction::commit_transaction() and transaction_future::then().
#define ASTM_FUTURE astm_detail::chainable_future

#define ASTM_FUNCTION std::function
#define ASTM_ASYNC astm_detail::async_wrapper
#define ASTM_MAKE_READY_FUTURE astm_detail::make_ready_future

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
