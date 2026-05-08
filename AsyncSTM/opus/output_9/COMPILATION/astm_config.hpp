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

// ---------------------------------------------------------------------------
// Legion-compatible future wrapper that supports .then() chaining required by
// the ASTM transaction commit mechanism.  Uses std::shared_future internally
// so that copies (needed by the STM bookkeeping) are cheap and valid.
// ---------------------------------------------------------------------------
namespace astm_detail {

template <typename T>
class legion_future;

template <>
class legion_future<void>
{
    std::shared_future<void> impl_;

public:
    legion_future() : impl_() {}

    // Allow implicit conversion from std::future<void>&& (returned by std::async)
    legion_future(std::future<void>&& f) : impl_(f.share()) {}

    legion_future(std::shared_future<void> f) : impl_(std::move(f)) {}

    legion_future(legion_future const&)            = default;
    legion_future(legion_future&&)                 = default;
    legion_future& operator=(legion_future const&) = default;
    legion_future& operator=(legion_future&&)      = default;

    void get()
    {
        if (impl_.valid())
            impl_.get();
    }

    // .then() continuation: waits for the current future, then runs f
    // std::bind result objects silently ignore extra arguments, matching
    // the way HPX's future::then passes the predecessor future.
    template <typename F>
    legion_future<void> then(F&& f)
    {
        auto shared = impl_;
        return legion_future<void>(
            std::async(std::launch::async,
                [shared,
                 func = std::decay_t<F>(std::forward<F>(f))]() mutable {
                    if (shared.valid())
                        shared.get();
                    func();
                }));
    }
};

inline legion_future<void> make_ready_future()
{
    std::promise<void> p;
    p.set_value();
    return legion_future<void>(p.get_future());
}

} // namespace astm_detail

// ---- Mutex / Lock (local synchronisation inside STM transactions) ---------
#define ASTM_MUTEX   std::mutex
#define ASTM_LOCK    std::unique_lock<std::mutex>

// ---- Futures / Async ------------------------------------------------------
#define ASTM_FUTURE            astm_detail::legion_future
#define ASTM_FUNCTION          std::function
#define ASTM_ASYNC             std::async
#define ASTM_MAKE_READY_FUTURE astm_detail::make_ready_future

// ---- Testing helpers ------------------------------------------------------
#define ASTM_TEST   assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// ---------------------------------------------------------------------------
// Newer ASTM forward-include guard
// ---------------------------------------------------------------------------
#if !defined(ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E)
#define ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E

#include "astm.hpp"

namespace astm
{
}

#endif
