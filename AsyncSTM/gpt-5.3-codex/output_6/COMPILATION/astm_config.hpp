////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// Legion runtime API (default mapper assumed by caller)
#include <legion.h>

#include <future>
#include <mutex>
#include <functional>
#include <type_traits>
#include <utility>
#include <cassert>

namespace astm
{

// Minimal future wrapper to preserve ASTM's HPX-style usage pattern:
// - default-construct as ready
// - get()
// - then(F) continuation returning same future type
template <typename T>
class legion_future;

template <>
class legion_future<void>
{
public:
    using future_type = std::shared_future<void>;

    legion_future()
      : fut_(make_ready_shared_())
    {}

    explicit legion_future(std::future<void>&& fut)
      : fut_(fut.share())
    {}

    explicit legion_future(future_type fut)
      : fut_(std::move(fut))
    {}

    void get()
    {
        fut_.get();
    }

    template <typename F>
    legion_future<void> then(F&& f)
    {
        auto prev = fut_;
        using fn_type = std::decay_t<F>;
        fn_type fn(std::forward<F>(f));

        auto next = std::async(std::launch::async,
            [prev, fn = std::move(fn)]() mutable {
                prev.get();
                fn();
            });

        return legion_future<void>(std::move(next));
    }

private:
    future_type fut_;

    static future_type make_ready_shared_()
    {
        std::promise<void> p;
        p.set_value();
        return p.get_future().share();
    }
};

inline legion_future<void> make_ready_future()
{
    return legion_future<void>();
}

// Host-side async helper used by ASTM_ASYNC.
// (Legion header is available for integration with Legion task launches elsewhere.)
template <typename F, typename... Args>
legion_future<void> legion_async(F&& f, Args&&... args)
{
    auto bound = std::bind(std::forward<F>(f), std::forward<Args>(args)...);

    auto fut = std::async(std::launch::async,
        [bound = std::move(bound)]() mutable {
            bound();
        });

    return legion_future<void>(std::move(fut));
}

} // namespace astm

#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<ASTM_MUTEX>
#define ASTM_FUTURE astm::legion_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC astm::legion_async
#define ASTM_MAKE_READY_FUTURE astm::make_ready_future

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
