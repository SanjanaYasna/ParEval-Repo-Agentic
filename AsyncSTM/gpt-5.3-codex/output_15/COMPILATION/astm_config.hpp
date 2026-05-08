////////////////////////////////////////////////////////////////////////////////
//  Legion-backed ASTM configuration
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E95

#include <legion.h>   // Legion runtime (default mapper assumed)

#include <cassert>
#include <functional>
#include <future>
#include <mutex>
#include <type_traits>
#include <utility>

namespace astm { namespace legion_backend {

// Lightweight future wrapper with HPX-like `.then(...)` chaining.
template <typename T>
class future
{
public:
    future() = default;
    explicit future(std::future<T>&& f) : fut_(f.share()) {}
    explicit future(std::shared_future<T> f) : fut_(std::move(f)) {}

    T get() { return fut_.get(); }
    void wait() const { fut_.wait(); }
    bool valid() const { return fut_.valid(); }

    template <typename F>
    auto then(F&& cont) const -> future<std::invoke_result_t<F, T>>
    {
        using R = std::invoke_result_t<F, T>;
        auto prev = fut_;
        auto chained = std::async(std::launch::async,
            [prev, fn = std::forward<F>(cont)]() mutable -> R {
                return fn(prev.get());
            });
        return future<R>(std::move(chained));
    }

private:
    std::shared_future<T> fut_;
};

template <>
class future<void>
{
public:
    future()
    {
        std::promise<void> p;
        p.set_value();
        fut_ = p.get_future().share();
    }

    explicit future(std::future<void>&& f) : fut_(f.share()) {}
    explicit future(std::shared_future<void> f) : fut_(std::move(f)) {}

    void get() { fut_.get(); }
    void wait() const { fut_.wait(); }
    bool valid() const { return fut_.valid(); }

    template <typename F>
    auto then(F&& cont) const -> future<std::invoke_result_t<F>>
    {
        using R = std::invoke_result_t<F>;
        auto prev = fut_;

        auto chained = std::async(std::launch::async,
            [prev, fn = std::forward<F>(cont)]() mutable -> R {
                prev.wait();
                if constexpr (std::is_void_v<R>) {
                    fn();
                    return;
                } else {
                    return fn();
                }
            });

        return future<R>(std::move(chained));
    }

private:
    std::shared_future<void> fut_;
};

template <typename F, typename... Args>
auto async(F&& f, Args&&... args)
    -> future<std::invoke_result_t<F, Args...>>
{
    using R = std::invoke_result_t<F, Args...>;
    auto sf = std::async(std::launch::async,
        std::forward<F>(f), std::forward<Args>(args)...);
    return future<R>(std::move(sf));
}

inline future<void> make_ready_future()
{
    std::promise<void> p;
    p.set_value();
    auto f = p.get_future();
    return future<void>(std::move(f));
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
