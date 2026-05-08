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
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <utility>

namespace astm { namespace detail {

template <typename T>
class legion_future;

// Forward declaration so legion_future<void>::then can chain through it.
template <typename F, typename... Args>
legion_future<void> legion_async(F&& f, Args&&... args);

template <>
class legion_future<void>
{
public:
    legion_future()
      : fut_(make_ready_shared_())
    {}

    explicit legion_future(std::future<void>&& f)
      : fut_(f.share())
    {}

    explicit legion_future(std::shared_future<void> f)
      : fut_(std::move(f))
    {}

    void get() const
    {
        fut_.get();
    }

    template <typename F>
    legion_future<void> then(F&& f) const
    {
        auto prev = fut_;
        return legion_async(
            [prev, fn = std::forward<F>(f)]() mutable {
                prev.get();
                std::invoke(fn);
            }
        );
    }

private:
    std::shared_future<void> fut_;

    static std::shared_future<void> make_ready_shared_()
    {
        std::promise<void> p;
        p.set_value();
        return p.get_future().share();
    }
};

template <typename F, typename... Args>
legion_future<void> legion_async(F&& f, Args&&... args)
{
    std::promise<void> p;
    auto sf = p.get_future().share();

    auto bound = std::bind(std::forward<F>(f), std::forward<Args>(args)...);

    std::thread(
        [pr = std::move(p), task = std::move(bound)]() mutable {
            try
            {
                task();
                pr.set_value();
            }
            catch (...)
            {
                pr.set_exception(std::current_exception());
            }
        }
    ).detach();

    return legion_future<void>(std::move(sf));
}

inline legion_future<void> make_ready_future()
{
    std::promise<void> p;
    p.set_value();
    return legion_future<void>(p.get_future().share());
}

}} // namespace astm::detail

#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<ASTM_MUTEX>
#define ASTM_FUTURE astm::detail::legion_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC astm::detail::legion_async
#define ASTM_MAKE_READY_FUTURE astm::detail::make_ready_future

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
