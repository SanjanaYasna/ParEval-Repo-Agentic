////////////////////////////////////////////////////////////////////////////////
//  Translated to Legion execution model (default mapper, header-only)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94

#include "astm_config.hpp"

#include <legion.h>

#include <list>
#include <set>
#include <map>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <utility>
#include <functional>
#include <unordered_map>

namespace astm
{

struct transaction;

//------------------------------------------------------------------------------
// Legion async glue (type-erased closure task)
//------------------------------------------------------------------------------
namespace detail
{
    inline constexpr Legion::TaskID ASTM_INTERNAL_ASYNC_TASK_ID = 0x7FFF1000;

    inline thread_local Legion::Runtime* tls_runtime = nullptr;
    inline thread_local Legion::Context  tls_context = Legion::Context();

    inline void bind_legion_context(Legion::Runtime* rt, Legion::Context ctx)
    {
        tls_runtime = rt;
        tls_context = ctx;
    }

    inline std::mutex closure_mtx;
    inline std::unordered_map<std::uint64_t, std::function<void()>> closures;
    inline std::atomic<std::uint64_t> next_closure_id{1};

    inline std::uint64_t store_closure(std::function<void()> fn)
    {
        const std::uint64_t id = next_closure_id.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(closure_mtx);
        closures.emplace(id, std::move(fn));
        return id;
    }

    inline std::function<void()> take_closure(std::uint64_t id)
    {
        std::lock_guard<std::mutex> lk(closure_mtx);
        auto it = closures.find(id);
        if (it == closures.end()) return {};
        std::function<void()> fn = std::move(it->second);
        closures.erase(it);
        return fn;
    }

    inline void astm_async_closure_task(
        const Legion::Task* task,
        const std::vector<Legion::PhysicalRegion>& /*regions*/,
        Legion::Context ctx,
        Legion::Runtime* runtime)
    {
        bind_legion_context(runtime, ctx);

        assert(task != nullptr);
        assert(task->arglen == sizeof(std::uint64_t));

        std::uint64_t id = 0;
        std::memcpy(&id, task->args, sizeof(std::uint64_t));

        auto fn = take_closure(id);
        if (fn) fn();
    }

    inline bool preregister_async_task()
    {
        Legion::TaskVariantRegistrar registrar(
            ASTM_INTERNAL_ASYNC_TASK_ID, "astm_async_closure_task");
        registrar.add_constraint(
            Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        registrar.set_leaf();

        Legion::Runtime::preregister_task_variant<astm_async_closure_task>(
            registrar, "astm_async_closure_task");
        return true;
    }

    // Ensure preregistration happens before Runtime::start.
    inline const bool async_task_registered = preregister_async_task();

    inline Legion::Future launch_async(
        std::function<void()> fn,
        Legion::Future const* dependency = nullptr)
    {
        (void)async_task_registered;

        // Fallback (e.g., unit tests before full Legion plumbing):
        if (tls_runtime == nullptr)
        {
            if (dependency && dependency->exists())
                dependency->get_void_result();
            fn();
            return Legion::Future();
        }

        const std::uint64_t id = store_closure(std::move(fn));
        Legion::TaskLauncher launcher(
            ASTM_INTERNAL_ASYNC_TASK_ID,
            Legion::TaskArgument(&id, sizeof(id)));

        if (dependency && dependency->exists())
            launcher.add_future(*dependency);

        return tls_runtime->execute_task(tls_context, launcher);
    }
} // namespace detail

// Call this once at the beginning of a Legion task that uses ASTM.
inline void bind_legion_context(Legion::Runtime* runtime, Legion::Context ctx)
{
    detail::bind_legion_context(runtime, ctx);
}

//------------------------------------------------------------------------------
// STM core
//------------------------------------------------------------------------------
struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;
    virtual void write(shared_var_base const&) = 0;
    virtual std::unique_lock<std::mutex> lock() const = 0;
    virtual bool operator==(shared_var_base const&) const = 0;
};

struct transaction_future
{
    using future_type = Legion::Future;

  private:
    transaction* trans_;
    future_type fut_;

  public:
    explicit transaction_future(transaction* trans)
      : trans_(trans), fut_()
    {}

    explicit transaction_future(transaction& trans)
      : trans_(&trans), fut_()
    {}

    template <typename F>
    void then(F f);

    void get()
    {
        if (fut_.exists())
            fut_.get_void_result();
    }
};

template <typename T>
struct shared_var : shared_var_base
{
    using future_type = Legion::Future;

    struct local_var
    {
      private:
        transaction* trans_;
        shared_var_base* var_;

      public:
        local_var(transaction* trans, shared_var_base* var)
          : trans_(trans), var_(var)
        {}

        T get() const;
        operator T const& () const;

        local_var& operator=(shared_var_base const& rhs);
        local_var& operator=(T const& rhs);

        template <typename F>
        void then(F f);
    };

  private:
    T data_;
    mutable std::mutex mtx_;

  public:
    future_type queue;

    shared_var() : data_(), mtx_(), queue() {}
    explicit shared_var(T const& t) : data_(t), mtx_(), queue() {}
    explicit shared_var(T&& t) : data_(std::move(t)), mtx_(), queue() {}
    shared_var(shared_var const& rhs) : data_(rhs.data_), mtx_(), queue() {}

    ~shared_var() {}

    shared_var_base* clone() const override
    {
        auto l = lock();
        (void)l;
        return new shared_var(data_);
    }

    T const& read() const
    {
        return data_;
    }

    void write(T const& rhs)
    {
        data_ = rhs;
    }

    void write(shared_var_base const& rhs) override
    {
        auto const* p = dynamic_cast<shared_var const*>(&rhs);
        assert(p != nullptr);
        data_ = p->read();
    }

    std::unique_lock<std::mutex> lock() const override
    {
        return std::unique_lock<std::mutex>(mtx_);
    }

    bool operator==(shared_var_base const& rhs) const override
    {
        auto const* p = dynamic_cast<shared_var const*>(&rhs);
        assert(p != nullptr);
        return data_ == p->read();
    }

    local_var get_local(transaction& trans)
    {
        return local_var(&trans, this);
    }
};

struct transaction
{
    std::list<
        std::pair<
            shared_var_base*,
            std::shared_ptr<shared_var_base>
        >
    > read_list;

    std::set<shared_var_base*> write_set;

    std::list<
        std::pair<
            Legion::Future*,
            std::function<void(transaction*)>
        >
    > async_list;

    std::map<
        shared_var_base*,
        std::shared_ptr<shared_var_base>
    > variables;

    void clear()
    {
        read_list.clear();
        write_set.clear();
        async_list.clear();
        variables.clear();
    }

    bool commit_transaction()
    {
        // 1) lock all vars in sorted order (map key order).
        std::list<std::unique_lock<std::mutex>> locks;
        for (auto const& var : variables)
        {
            assert(var.first != nullptr);
            locks.push_back(var.first->lock());
        }

        // 2) validate reads.
        for (auto const& var : read_list)
        {
            assert(var.first != nullptr);
            if (!(*var.first == *var.second))
            {
                clear();
                return false;
            }
        }

        // 3) apply writes.
        for (shared_var_base* var : write_set)
        {
            assert(var != nullptr);
            auto it = variables.find(var);
            assert(it != variables.end());
            var->write(*it->second);
        }

        // 4) schedule async ops.
        for (auto& op : async_list)
        {
            if (op.first == nullptr)
            {
                detail::launch_async([fn = op.second, this]() { fn(this); }, nullptr);
            }
            else
            {
                Legion::Future dep = *op.first;
                *op.first = detail::launch_async(
                    [fn = op.second, this]() { fn(this); },
                    dep.exists() ? &dep : nullptr);
            }
        }

        // 5) unlock by RAII.
        return true;
    }

    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != nullptr);

        auto result = variables.insert({var, std::shared_ptr<shared_var_base>()});

        if (result.second)
        {
            // First access: clone current shared value into transactional state.
            result.first->second.reset(var->clone());

            // Snapshot read value for validation (separate clone).
            read_list.push_back({
                var,
                std::shared_ptr<shared_var_base>(result.first->second->clone())
            });
        }

        return *result.first->second;
    }

    void write(shared_var_base* var, shared_var_base const& value)
    {
        assert(var != nullptr);

        auto result = variables.insert({
            var,
            std::shared_ptr<shared_var_base>(value.clone())
        });

        if (!result.second)
        {
            result.first->second.reset(value.clone());
        }

        write_set.insert(var);
    }

    void then(Legion::Future* fut, std::function<void(transaction*)> F)
    {
        async_list.push_back({fut, std::move(F)});
    }
};

template <typename T>
shared_var<T>::local_var::operator T const& () const
{
    auto const* p = dynamic_cast<shared_var const*>(&trans_->read(var_));
    assert(p != nullptr);
    return p->read();
}

template <typename T>
T shared_var<T>::local_var::get() const
{
    auto const* p = dynamic_cast<shared_var const*>(&trans_->read(var_));
    assert(p != nullptr);
    return p->read();
}

template <typename T>
typename shared_var<T>::local_var&
shared_var<T>::local_var::operator=(shared_var_base const& rhs)
{
    trans_->write(var_, rhs);
    return *this;
}

template <typename T>
typename shared_var<T>::local_var&
shared_var<T>::local_var::operator=(T const& rhs)
{
    shared_var tmp(rhs);
    trans_->write(var_, tmp);
    return *this;
}

template <typename T>
template <typename F>
void shared_var<T>::local_var::then(F f)
{
    assert(trans_ != nullptr);
    auto* concrete = dynamic_cast<shared_var<T>*>(var_);
    assert(concrete != nullptr);
    trans_->then(&concrete->queue, std::function<void(transaction*)>(std::move(f)));
}

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_ != nullptr);
    trans_->then(&fut_, std::function<void(transaction*)>(std::move(f)));
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
