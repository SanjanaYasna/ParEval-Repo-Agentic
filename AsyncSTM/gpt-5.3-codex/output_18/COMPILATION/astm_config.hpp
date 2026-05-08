////////////////////////////////////////////////////////////////////////////////
//  Translated ASTM configuration for Legion-based builds (default mapper).
//  HPX dependencies removed; this header provides a small future/async shim
//  compatible with ASTM's existing API surface.
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

#include <legion.h>

#include <cassert>
#include <functional>
#include <future>
#include <mutex>
#include <type_traits>
#include <utility>

namespace astm
{

// Minimal continuation-capable future used by ASTM.
// NOTE: ASTM currently only requires FUTURE<void>.
template <typename T>
class legion_future;

template <>
class legion_future<void>
{
private:
    std::shared_future<void> fut_;

    static std::shared_future<void> make_ready_shared_future()
    {
        std::promise<void> p;
        p.set_value();
        return p.get_future().share();
    }

public:
    legion_future()
      : fut_(make_ready_shared_future())
    {}

    explicit legion_future(std::future<void>&& fut)
      : fut_(fut.share())
    {}

    explicit legion_future(std::shared_future<void> fut)
      : fut_(std::move(fut))
    {}

    void get()
    {
        fut_.get();
    }

    template <typename F>
    legion_future<void> then(F&& f) const
    {
        auto prev = fut_;
        auto cont = std::async(
            std::launch::async,
            [prev, fn = std::forward<F>(f)]() mutable {
                prev.get();          // propagate any exception from predecessor
                std::invoke(fn);     // run continuation
            });

        return legion_future<void>(std::move(cont));
    }
};

inline legion_future<void> make_ready_future()
{
    return legion_future<void>();
}

template <typename F, typename... Args>
legion_future<void> legion_async(F&& f, Args&&... args)
{
    using result_type = std::invoke_result_t<F, Args...>;
    static_assert(std::is_void<result_type>::value,
        "ASTM async operations are expected to return void.");

    auto fut = std::async(
        std::launch::async,
        std::forward<F>(f),
        std::forward<Args>(args)...);

    return legion_future<void>(std::move(fut));
}

} // namespace astm

// ASTM backend definitions for Legion translation.
#define ASTM_MUTEX std::mutex
#define ASTM_LOCK std::unique_lock<ASTM_MUTEX>
#define ASTM_FUTURE ::astm::legion_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC ::astm::legion_async
#define ASTM_MAKE_READY_FUTURE ::astm::make_ready_future

#define ASTM_TEST assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95
