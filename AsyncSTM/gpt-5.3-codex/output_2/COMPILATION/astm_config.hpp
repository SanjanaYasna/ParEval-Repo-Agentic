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

namespace astm
{
    inline std::future<void> make_ready_future()
    {
        std::promise<void> p;
        p.set_value();
        return p.get_future();
    }
} // namespace astm

#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<ASTM_MUTEX>
#define ASTM_FUTURE std::future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC std::async
#define ASTM_MAKE_READY_FUTURE astm::make_ready_future()

#define ASTM_TEST assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95
