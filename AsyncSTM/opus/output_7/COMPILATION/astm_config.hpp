////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// Legion execution model
#include "legion.h"

#include <future>
#include <thread>
#include <mutex>
#include <functional>
#include <fstream>
#include <cassert>

using namespace Legion;

// Internal STM synchronization uses std:: primitives.
// Legion manages inter-task synchronization through logical regions
// and task dependencies via the runtime, but the STM commit protocol
// (optimistic locking, future chaining) operates within a task's
// local execution context, so std:: concurrency types are appropriate.
#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<std::mutex>
#define ASTM_FUTURE std::future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC std::async

// Note: std:: has no direct make_ready_future; launch a trivial async as a workaround.
#define ASTM_MAKE_READY_FUTURE std::async(std::launch::async, [](){})

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
