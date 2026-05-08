////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// Legion header – provides Runtime, Task, Future, LogicalRegion, etc.
#include "legion.h"

// Standard library primitives used for the internal STM synchronization
// and async chaining (Legion::Future does not support .then() directly,
// so std::future / std::async are used for the STM internals).
#include <future>
#include <thread>
#include <mutex>
#include <functional>
#include <cassert>
#include <fstream>

// ---------------------------------------------------------------------------
// Mutex / Lock – used inside transaction::commit_transaction() to acquire
// exclusive access to shared variables.  These are intra-process locks and
// work fine inside a Legion task.
// ---------------------------------------------------------------------------
#define ASTM_MUTEX   std::mutex
#define ASTM_LOCK    std::unique_lock<std::mutex>

// ---------------------------------------------------------------------------
// Future / Async – used by the STM library for fire-and-forget semantics,
// transaction_future::then() chaining, and the shared_var queue.
//
// We use std::future / std::async because:
//   1. Legion::Future<T> uses .get_result<T>() rather than .get().
//   2. Legion::Future does not expose a .then() continuation interface.
//   3. The STM layer only needs intra-process async; true inter-node
//      parallelism is expressed through Legion task launches in the
//      application code (the .cpp files).
// ---------------------------------------------------------------------------
#define ASTM_FUTURE              std::future
#define ASTM_FUNCTION            std::function
#define ASTM_ASYNC               std::async

// Note: std:: has no make_ready_future; create an immediately-ready future.
#define ASTM_MAKE_READY_FUTURE   std::async(std::launch::deferred, [](){})

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------
#define ASTM_TEST   assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

// ---------------------------------------------------------------------------
// Second translation-unit guard (mirrors the original structure).
// ---------------------------------------------------------------------------
#if !defined(ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E)
#define ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E

#include "astm.hpp"

namespace astm
{
}

#endif // ASTM_65EB3381_65AE_4527_A4E5_A87D5014A33E
