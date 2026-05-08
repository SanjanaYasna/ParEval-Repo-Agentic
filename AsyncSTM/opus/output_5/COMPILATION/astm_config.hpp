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
using namespace Legion;

#include <future>
#include <thread>
#include <mutex>
#include <functional>

// STM-internal synchronization: std::mutex is appropriate since the
// transactional mechanism operates within shared memory on a single node.
#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<std::mutex>

// Legion's Future does not support .then() or the .get() interface that the
// STM library's commit and async-chaining logic requires, so we use
// std::future / std::async for the internal STM plumbing.  Legion tasks and
// regions are used at the application level (in the .cpp drivers).
#define ASTM_FUTURE std::future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC std::async

// Note: std:: has no make_ready_future; create an immediately-ready future.
#define ASTM_MAKE_READY_FUTURE std::async(std::launch::deferred, [](){})

#include <assert.h>

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
