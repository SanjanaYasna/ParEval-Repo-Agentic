////////////////////////////////////////////////////////////////////////////////
//  Translated to Legion execution model (default mapper)
//  ASTM configuration for Legion-backed async/future utilities.
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

#include "legion.h"

#include <cassert>
#include <cstdint>
#include <future>
#include <functional>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace astm { namespace legion_backend {

enum : Legion::TaskID {
    ASTM_INTERNAL_ASYNC_TASK_ID = 0x5A7A0001 // unique task id for internal async calls
};

inline thread_local Legion::Runtime* tls_runtime = nullptr;
inline thread_local Legion::Context tls_context = Legion::Context();

inline void bind_runtime_context(Legion::Runtime* runtime, Legion::Context context)
{
    tls_runtime = runtime;
    tls_context = context;
}

inline bool has_bound_runtime_context()
{
    return (tls_runtime != nullptr) && (tls_context != Legion::Context());
}

inline std::mutex pending_mutex;
inline std::unordered_map<std::uint64_t, std::function<void()>> pending_calls;
inline std::uint64_t pending_next_id = 1;

inline std::uint64_t enqueue_call(std::function<void()> fn)
{
    std::lock_guard<std::mutex> lk(pending_mutex);
    const std::uint64_t id = pending_next_id++;
    pending_calls.emplace(id, std::move(fn));
    return id;
}

inline std::function<void()> dequeue_call(std::uint64_t id)
{
    std::lock_guard<std::mutex> lk(pending_mutex);
    auto it = pending_calls.find(id);
    assert(it != pending_calls.end());
    std::function<void()> fn = std::move(it->second);
    pending_calls.erase(it);
    return fn;
}

inline void astm_internal_async_task(
    Legion::Task const* task,
    std::vector<Legion::PhysicalRegion> const&,
    Legion::Context ctx,
    Legion::Runtime* runtime)
{
    assert(task->arglen == sizeof(std::uint64_t));
    const auto id = *static_cast<std::uint64_t const*>(task->args);

    auto fn = dequeue_call(id);

    Legion::Runtime* prev_runtime = tls_runtime;
    Legion::Context prev_context = tls_context;
    bind_runtime_context(runtime, ctx);

    fn();

    tls_runtime = prev_runtime;
    tls_context = prev_context;
}

inline bool astm_internal_async_registered = []() {
    Legion::TaskVariantRegistrar registrar(
        ASTM_INTERNAL_ASYNC_TASK_ID, "astm_internal_async_task");
    registrar.add_constraint(
        Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    Legion::Runtime::preregister_task_variant<astm_internal_async_task>(
        registrar, "astm_internal_async_task");
    return true;
}();

template <typename T>
class future
{
    static_assert(std::is_same<T, void>::value,
        "astm::legion_backend::future currently supports only void.");
};

template <>
class future<void>
{
public:
    enum class kind_t { ready, legion, host };

    future() = default;

    explicit future(Legion::Future f)
      : kind_(kind_t::legion), legion_future_(std::move(f))
    {}

    explicit future(std::shared_future<void> f)
      : kind_(kind_t::host), host_future_(std::move(f))
    {}

    void get() const
    {
        if (kind_ == kind_t::legion)
        {
            if (legion_future_.exists()) legion_future_.get_void_result();
        }
        else if (kind_ == kind_t::host)
        {
            host_future_.wait();
            host_future_.get();
        }
        // ready => no-op
    }

    template <typename F>
    future<void> then(F&& f) const;

private:
    kind_t kind_ = kind_t::ready;
    Legion::Future legion_future_;
    std::shared_future<void> host_future_;
};

inline future<void> launch_host(std::function<void()> fn)
{
    auto sf = std::async(std::launch::async, [f = std::move(fn)]() mutable {
        f();
    }).share();
    return future<void>(std::move(sf));
}

inline future<void> launch_legion(std::function<void()> fn)
{
    (void)astm_internal_async_registered;
    assert(has_bound_runtime_context());

    const std::uint64_t id = enqueue_call(std::move(fn));
    Legion::TaskLauncher launcher(
        ASTM_INTERNAL_ASYNC_TASK_ID,
        Legion::TaskArgument(&id, sizeof(id)));
    Legion::Future f = tls_runtime->execute_task(tls_context, launcher);
    return future<void>(std::move(f));
}

template <typename F, typename... Args>
future<void> async(F&& f, Args&&... args)
{
    auto bound = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    std::function<void()> fn = [b = std::move(bound)]() mutable { b(); };

    if (has_bound_runtime_context())
        return launch_legion(std::move(fn));
    return launch_host(std::move(fn));
}

template <typename F>
future<void> future<void>::then(F&& f) const
{
    auto prev = *this;
    std::function<void()> fn = std::forward<F>(f);

    return async([prev, fn = std::move(fn)]() mutable {
        prev.get();
        fn();
    });
}

inline future<void> make_ready_future()
{
    return future<void>();
}

}} // namespace astm::legion_backend

#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<ASTM_MUTEX>
#define ASTM_FUTURE astm::legion_backend::future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC astm::legion_backend::async
#define ASTM_MAKE_READY_FUTURE astm::legion_backend::make_ready_future

#define ASTM_TEST assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95
