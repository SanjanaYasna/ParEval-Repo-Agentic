////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94

#include "astm_config.hpp"

#include <legion.h>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace astm
{

class future_void;

namespace detail
{
    struct runtime_binding
    {
        Legion::Runtime* runtime{nullptr};
        Legion::Context context{nullptr};
    };

    inline thread_local runtime_binding tls_runtime_binding{};

    future_void launch_async(ASTM_FUNCTION<void()> fn, future_void const* dependency);
} // namespace detail

inline void bind_legion_context(Legion::Runtime* runtime, Legion::Context context)
{
    detail::tls_runtime_binding.runtime = runtime;
    detail::tls_runtime_binding.context = context;
}

inline void clear_legion_context()
{
    detail::tls_runtime_binding.runtime = nullptr;
    detail::tls_runtime_binding.context = nullptr;
}

class future_void
{
  public:
    future_void() = default;

    explicit future_void(Legion::Future f)
      : has_future_(f.exists())
      , future_(std::move(f))
    {}

    void get() const
    {
        if (has_future_ && future_.exists())
            future_.get_void_result();
    }

    bool valid() const
    {
        return has_future_ && future_.exists();
    }

    Legion::Future const& raw_future() const
    {
        return future_;
    }

    template <typename F>
    future_void then(F&& f) const
    {
        ASTM_FUNCTION<void()> fn(std::forward<F>(f));
        return detail::launch_async(std::move(fn), this);
    }

  private:
    bool has_future_{false};
    Legion::Future future_{};
};

namespace detail
{
    inline constexpr Legion::TaskID ASTM_INTERNAL_ASYNC_TASK_ID = 1000001;

    inline std::mutex callback_mutex;
    inline std::unordered_map<std::uint64_t, ASTM_FUNCTION<void()>> callbacks;
    inline std::atomic<std::uint64_t> next_callback_id{1};

    inline void astm_internal_async_task(
        Legion::Task const* task,
        std::vector<Legion::PhysicalRegion> const&,
        Legion::Context,
        Legion::Runtime*)
    {
        assert(task != nullptr);
        assert(task->arglen == static_cast<int>(sizeof(std::uint64_t)));

        std::uint64_t const callback_id =
            *reinterpret_cast<std::uint64_t const*>(task->args);

        ASTM_FUNCTION<void()> fn;
        {
            std::lock_guard<std::mutex> lock(callback_mutex);
            auto it = callbacks.find(callback_id);
            if (it != callbacks.end())
            {
                fn = std::move(it->second);
                callbacks.erase(it);
            }
        }

        if (fn)
            fn();
    }

    struct task_registrar
    {
        task_registrar()
        {
            Legion::TaskVariantRegistrar registrar(
                ASTM_INTERNAL_ASYNC_TASK_ID, "astm_internal_async_task");
            registrar.add_constraint(
                Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
            Legion::Runtime::preregister_task_variant<astm_internal_async_task>(
                registrar, "astm_internal_async_task");
        }
    };

    inline task_registrar registrar_instance{};

    inline future_void launch_async(ASTM_FUNCTION<void()> fn, future_void const* dependency)
    {
        // Ensure registration side effect is retained.
        (void)registrar_instance;

        runtime_binding const& binding = tls_runtime_binding;

        // Fallback path: run synchronously if no Legion runtime/context is bound.
        if (binding.runtime == nullptr || binding.context == nullptr)
        {
            if (dependency != nullptr)
                dependency->get();
            if (fn)
                fn();
            return future_void{};
        }

        std::uint64_t const callback_id =
            next_callback_id.fetch_add(1, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lock(callback_mutex);
            callbacks.emplace(callback_id, std::move(fn));
        }

        Legion::TaskLauncher launcher(
            ASTM_INTERNAL_ASYNC_TASK_ID,
            Legion::TaskArgument(&callback_id, sizeof(callback_id)));

        if (dependency != nullptr && dependency->valid())
            launcher.add_future(dependency->raw_future());

        Legion::Future f = binding.runtime->execute_task(binding.context, launcher);
        return future_void(std::move(f));
    }

} // namespace detail

struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;
    virtual void write(shared_var_base const&) = 0;
    virtual ASTM_LOCK lock() const = 0;
    virtual bool operator==(shared_var_base const&) const = 0;
};

struct transaction;

struct transaction_future
{
    using future_type = future_void;

  private:
    transaction* trans_;
    future_type fut_;

  public:
    explicit transaction_future(transaction* trans)
      : trans_(trans)
      , fut_()
    {}

    explicit transaction_future(transaction& trans)
      : trans_(&trans)
      , fut_()
    {}

    template <typename F>
    void then(F f);

    void get()
    {
        fut_.get();
    }
};

template <typename T>
struct shared_var : shared_var_base
{
    using future_type = future_void;

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
        void then(F f);
    };

  private:
    T data_;
    mutable ASTM_MUTEX mtx_;

  public:
    future_type queue;

    shared_var()
      : data_()
      , mtx_()
      , queue()
    {}

    explicit shared_var(T const& t)
      : data_(t)
      , mtx_()
      , queue()
    {}

    explicit shared_var(T&& t)
      : data_(std::move(t))
      , mtx_()
      , queue()
    {}

    shared_var(shared_var const& rhs)
      : data_(rhs.data_)
      , mtx_()
      , queue()
    {}

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
        data_ = dynamic_cast<shared_var const*>(&rhs)->read();
    }

    ASTM_LOCK lock() const override
    {
        return ASTM_LOCK(mtx_);
    }

    bool operator==(shared_var_base const& rhs) const override
    {
        return data_ == dynamic_cast<shared_var const*>(&rhs)->read();
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

    std::list<std::pair<future_void*, ASTM_FUNCTION<void(transaction*)>>> async_list;

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
        std::list<ASTM_LOCK> locks;

        for (auto const& var : variables)
        {
            assert(var.first != nullptr);
            locks.push_back(var.first->lock());
        }

        for (auto const& var : read_list)
        {
            assert(var.first != nullptr);

            if (!((*var.first) == (*var.second)))
            {
                clear();
                return false;
            }
        }

        for (shared_var_base* var : write_set)
        {
            assert(var != nullptr);

            auto it = variables.find(var);
            assert(it != variables.end());

            var->write(*it->second);
        }

        for (auto& op : async_list)
        {
            if (op.first == nullptr)
            {
                detail::launch_async(
                    [fn = op.second, this]() { fn(this); }, nullptr);
            }
            else
            {
                *op.first =
                    op.first->then([fn = op.second, this]() { fn(this); });
            }
        }

        return true;
    }

    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != nullptr);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base>> entry(var, nullptr);
        auto result = variables.insert(entry);

        if (result.second)
        {
            result.first->second.reset(var->clone());
            read_list.push_back(*result.first);
            return *result.first->second;
        }

        return *result.first->second;
    }

    void write(shared_var_base* var, shared_var_base const& value)
    {
        assert(var != nullptr);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base>> entry(
            var, std::shared_ptr<shared_var_base>(value.clone()));

        auto result = variables.insert(entry);

        if (!result.second)
            result.first->second = entry.second;

        write_set.insert(var);
    }

    void then(future_void* fut, ASTM_FUNCTION<void(transaction*)> F)
    {
        async_list.emplace_back(fut, std::move(F));
    }
};

template <typename T>
shared_var<T>::local_var::operator T const&() const
{
    return dynamic_cast<shared_var const*>(&trans_->read(var_))->read();
}

template <typename T>
T shared_var<T>::local_var::get() const
{
    return dynamic_cast<shared_var const*>(&trans_->read(var_))->read();
}

template <typename T>
typename shared_var<T>::local_var& shared_var<T>::local_var::operator=(
    shared_var_base const& rhs)
{
    trans_->write(var_, rhs);
    return *this;
}

template <typename T>
typename shared_var<T>::local_var& shared_var<T>::local_var::operator=(
    T const& rhs)
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

    trans_->then(&concrete->queue, ASTM_FUNCTION<void(transaction*)>(std::move(f)));
}

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_ != nullptr);
    trans_->then(&fut_, ASTM_FUNCTION<void(transaction*)>(std::move(f)));
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
