////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0.
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94

#include "astm_config.hpp"

#include <cassert>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <future>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(ASTM_LEGION)
#include "legion.h"
#endif

namespace astm
{

struct transaction;

namespace detail
{
    using continuation_future = std::shared_future<void>;

    inline continuation_future make_ready_future()
    {
        std::promise<void> p;
        p.set_value();
        return p.get_future().share();
    }

    template <typename>
    inline constexpr bool dependent_false_v = false;

    template <typename F>
    std::function<void(transaction*)> to_tx_function(F&& f)
    {
        using fn_t = std::decay_t<F>;
        return [fn = fn_t(std::forward<F>(f))](transaction* t) mutable {
            if constexpr (std::is_invocable_v<fn_t, transaction*>)
                fn(t);
            else if constexpr (std::is_invocable_v<fn_t>)
                fn();
            else
                static_assert(dependent_false_v<fn_t>,
                    "Continuation must be invocable as f() or f(transaction*)");
        };
    }

#if defined(ASTM_LEGION)
    // Bind these from inside a Legion task before using astm::transaction async ops.
    inline thread_local Legion::Runtime* tls_runtime = nullptr;
    inline thread_local Legion::Context  tls_context = Legion::Context();

    inline std::mutex callback_registry_mtx;
    inline std::unordered_map<uint64_t, std::function<void()>> callback_registry;
    inline std::atomic<uint64_t> next_token{1};

    inline constexpr Legion::TaskID ASTM_INTERNAL_ASYNC_TASK_ID = 0x5A7A5701;

    inline void astm_internal_async_task(
        const Legion::Task* task,
        const std::vector<Legion::PhysicalRegion>&,
        Legion::Context,
        Legion::Runtime*)
    {
        uint64_t token = 0;
        if (task != nullptr && task->args != nullptr && task->arglen == sizeof(uint64_t))
            std::memcpy(&token, task->args, sizeof(uint64_t));

        std::function<void()> fn;
        {
            std::lock_guard<std::mutex> lk(callback_registry_mtx);
            auto it = callback_registry.find(token);
            if (it != callback_registry.end())
            {
                fn = std::move(it->second);
                callback_registry.erase(it);
            }
        }

        if (fn)
            fn();
    }

    inline void register_internal_async_task()
    {
        Legion::TaskVariantRegistrar registrar(
            ASTM_INTERNAL_ASYNC_TASK_ID, "astm_internal_async_task");
        registrar.add_constraint(
            Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<astm_internal_async_task>(
            registrar, "astm_internal_async_task");
    }

    struct registration_bootstrap
    {
        registration_bootstrap() { register_internal_async_task(); }
    };

    inline registration_bootstrap registration_bootstrap_instance{};
#endif

    inline continuation_future launch_async(
        std::function<void(transaction*)> fn,
        transaction* trans)
    {
#if defined(ASTM_LEGION)
        if (tls_runtime != nullptr)
        {
            const uint64_t token =
                next_token.fetch_add(1, std::memory_order_relaxed);

            {
                std::lock_guard<std::mutex> lk(callback_registry_mtx);
                callback_registry.emplace(
                    token, [f = std::move(fn), trans]() mutable { f(trans); });
            }

            Legion::TaskLauncher launcher(
                ASTM_INTERNAL_ASYNC_TASK_ID,
                Legion::TaskArgument(&token, sizeof(token)));

            Legion::Future lf = tls_runtime->execute_task(tls_context, launcher);

            return std::async(std::launch::async, [lf]() mutable {
                lf.get_void_result();
            }).share();
        }
#endif
        return std::async(std::launch::async, [f = std::move(fn), trans]() mutable {
            f(trans);
        }).share();
    }

} // namespace detail

#if defined(ASTM_LEGION)
inline void bind_legion_context(Legion::Runtime* runtime, Legion::Context context)
{
    detail::tls_runtime = runtime;
    detail::tls_context = context;
    legion_backend::bind_runtime_context(runtime, context);
}

inline void unbind_legion_context()
{
    detail::tls_runtime = nullptr;
    detail::tls_context = Legion::Context();
    legion_backend::bind_runtime_context(nullptr, Legion::Context());
}
#endif

struct shared_var_base
{
    using lock_type = std::unique_lock<std::mutex>;

    virtual ~shared_var_base() = default;

    virtual shared_var_base* clone() const = 0;
    virtual void write(shared_var_base const&) = 0;
    virtual lock_type lock() const = 0;
    virtual bool operator==(shared_var_base const&) const = 0;
};

struct transaction_future
{
    using future_type = detail::continuation_future;

  private:
    transaction* trans_;
    future_type fut_;

  public:
    explicit transaction_future(transaction* trans)
      : trans_(trans)
      , fut_(detail::make_ready_future())
    {}

    explicit transaction_future(transaction& trans)
      : trans_(&trans)
      , fut_(detail::make_ready_future())
    {}

    template <typename F>
    void then(F&& f);

    void get()
    {
        if (fut_.valid())
            fut_.wait();
    }
};

template <typename T>
struct shared_var : shared_var_base
{
    using future_type = detail::continuation_future;

    struct local_var
    {
      private:
        transaction* trans_;
        shared_var_base* var_;

      public:
        local_var(transaction* trans, shared_var_base* var)
          : trans_(trans)
          , var_(var)
        {}

        T get() const;
        operator T const&() const;

        local_var& operator=(shared_var_base const& rhs);
        local_var& operator=(T const& rhs);

        template <typename F>
        void then(F&& f);
    };

  private:
    T data_;
    mutable std::mutex mtx_;

  public:
    future_type queue;

    shared_var()
      : data_()
      , mtx_()
      , queue(detail::make_ready_future())
    {}

    explicit shared_var(T const& t)
      : data_(t)
      , mtx_()
      , queue(detail::make_ready_future())
    {}

    explicit shared_var(T&& t)
      : data_(std::move(t))
      , mtx_()
      , queue(detail::make_ready_future())
    {}

    shared_var(shared_var const& rhs)
      : data_(rhs.data_)
      , mtx_()
      , queue(detail::make_ready_future())
    {}

    ~shared_var() override = default;

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

    lock_type lock() const override
    {
        return lock_type(mtx_);
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
    std::list<std::pair<shared_var_base*, std::shared_ptr<shared_var_base>>> read_list;
    std::set<shared_var_base*> write_set;

    std::list<
        std::pair<detail::continuation_future*, std::function<void(transaction*)>>
    > async_list;

    std::map<shared_var_base*, std::shared_ptr<shared_var_base>> variables;

    void clear()
    {
        read_list.clear();
        write_set.clear();
        async_list.clear();
        variables.clear();
    }

    bool commit_transaction()
    {
        // 1) Lock all variables (ordered by pointer because std::map is ordered)
        std::list<shared_var_base::lock_type> locks;
        for (auto const& var : variables)
        {
            assert(var.first != nullptr);
            locks.emplace_back(var.first->lock());
        }

        // 2) Validate read set
        for (auto const& var : read_list)
        {
            assert(var.first != nullptr);
            if (!((*var.first) == (*var.second)))
            {
                clear();
                return false;
            }
        }

        // 3) Apply writes
        for (shared_var_base* var : write_set)
        {
            assert(var != nullptr);
            auto it = variables.find(var);
            assert(it != variables.end());
            var->write(*it->second);
        }

        // 4) Launch async operations
        for (auto& op : async_list)
        {
            if (op.first == nullptr)
            {
                (void)detail::launch_async(op.second, this);
            }
            else
            {
                detail::continuation_future prev = *op.first;
                *op.first = std::async(std::launch::async,
                    [prev, fn = op.second, this]() mutable {
                        if (prev.valid())
                            prev.wait();
                        auto cur = detail::launch_async(std::move(fn), this);
                        if (cur.valid())
                            cur.wait();
                    }).share();
            }
        }

        // 5) Locks released by RAII
        return true;
    }

    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != nullptr);

        auto result = variables.emplace(var, std::shared_ptr<shared_var_base>{});
        if (result.second)
        {
            result.first->second.reset(var->clone());
            read_list.emplace_back(result.first->first, result.first->second);
        }

        return *result.first->second;
    }

    void write(shared_var_base* var, shared_var_base const& value)
    {
        assert(var != nullptr);

        auto result = variables.emplace(
            var, std::shared_ptr<shared_var_base>(value.clone()));

        if (!result.second)
        {
            assert(result.first->second);
            result.first->second->write(value);
        }

        write_set.insert(var);
    }

    void then(detail::continuation_future* fut, std::function<void(transaction*)> F)
    {
        async_list.emplace_back(fut, std::move(F));
    }
};

template <typename T>
shared_var<T>::local_var::operator T const&() const
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
void shared_var<T>::local_var::then(F&& f)
{
    assert(trans_ != nullptr);
    auto* owner = dynamic_cast<shared_var*>(var_);
    assert(owner != nullptr);
    trans_->then(&owner->queue, detail::to_tx_function(std::forward<F>(f)));
}

template <typename F>
void transaction_future::then(F&& f)
{
    assert(trans_ != nullptr);
    trans_->then(&fut_, detail::to_tx_function(std::forward<F>(f)));
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
