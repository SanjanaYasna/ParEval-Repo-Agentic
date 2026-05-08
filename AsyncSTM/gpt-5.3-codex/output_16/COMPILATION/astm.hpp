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
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>
#include <functional>

namespace astm
{

struct transaction;

namespace detail
{
    constexpr Legion::TaskID ASTM_INTERNAL_ASYNC_TASK_ID = 0x00A57A57;

    struct async_payload
    {
        std::uint64_t op_id;
    };

    inline std::mutex& async_registry_mutex()
    {
        static std::mutex m;
        return m;
    }

    inline std::unordered_map<std::uint64_t, std::function<void()>>& async_registry()
    {
        static std::unordered_map<std::uint64_t, std::function<void()>> reg;
        return reg;
    }

    inline std::atomic<std::uint64_t>& next_async_id()
    {
        static std::atomic<std::uint64_t> id{1};
        return id;
    }

    inline void astm_async_task(
        const Legion::Task* task,
        const std::vector<Legion::PhysicalRegion>&,
        Legion::Context,
        Legion::Runtime*)
    {
        assert(task != nullptr);
        assert(task->arglen == sizeof(async_payload));

        const auto* payload = static_cast<const async_payload*>(task->args);

        std::function<void()> fn;
        {
            std::lock_guard<std::mutex> lk(async_registry_mutex());
            auto it = async_registry().find(payload->op_id);
            if (it != async_registry().end())
            {
                fn = std::move(it->second);
                async_registry().erase(it);
            }
        }

        if (fn) fn();
    }

    struct async_task_registrar
    {
        async_task_registrar()
        {
            Legion::TaskVariantRegistrar registrar(
                ASTM_INTERNAL_ASYNC_TASK_ID, "astm_internal_async_task");
            registrar.add_constraint(
                Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
            registrar.set_leaf();

            Legion::Runtime::preregister_task_variant<astm_async_task>(
                registrar, "astm_internal_async_task");
        }
    };

    inline async_task_registrar astm_async_task_registrar_instance{};

    inline Legion::Future launch_async(
        ASTM_FUNCTION<void(transaction*)> fn,
        transaction* trans,
        const Legion::Future* dependency = nullptr)
    {
        std::uint64_t id = next_async_id().fetch_add(1, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lk(async_registry_mutex());
            async_registry().emplace(
                id,
                [f = std::move(fn), trans]() mutable {
                    if (f) f(trans);
                });
        }

        Legion::Runtime* runtime = Legion::Runtime::get_runtime();
        Legion::Context ctx = Legion::Runtime::get_context();

        // Fallback path: no active Legion context.
        if (runtime == nullptr || ctx == nullptr)
        {
            std::function<void()> local_fn;
            {
                std::lock_guard<std::mutex> lk(async_registry_mutex());
                auto it = async_registry().find(id);
                if (it != async_registry().end())
                {
                    local_fn = std::move(it->second);
                    async_registry().erase(it);
                }
            }
            if (local_fn) local_fn();
            return Legion::Future();
        }

        async_payload payload{id};
        Legion::TaskLauncher launcher(
            ASTM_INTERNAL_ASYNC_TASK_ID,
            Legion::TaskArgument(&payload, sizeof(payload)));

        if (dependency != nullptr && dependency->exists())
            launcher.add_future(*dependency);

        return runtime->execute_task(ctx, launcher);
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
    mutable ASTM_MUTEX mtx_;

  public:
    mutable future_type queue;

    shared_var()
      : data_(), mtx_(), queue()
    {}

    explicit shared_var(T const& t)
      : data_(t), mtx_(), queue()
    {}

    explicit shared_var(T&& t)
      : data_(std::move(t)), mtx_(), queue()
    {}

    shared_var(shared_var const& rhs)
      : data_(rhs.data_), mtx_(), queue()
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
    using future_type = Legion::Future;

    std::list<
        std::pair<
            shared_var_base*,
            std::shared_ptr<shared_var_base>
        >
    > read_list;

    std::set<shared_var_base*> write_set;

    std::list<
        std::pair<
            future_type*,
            ASTM_FUNCTION<void(transaction*)>
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
                detail::launch_async(std::move(op.second), this, nullptr);
            }
            else
            {
                Legion::Future next =
                    detail::launch_async(
                        std::move(op.second),
                        this,
                        (op.first->exists() ? op.first : nullptr));
                *op.first = next;
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

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base>>
            entry(var, std::shared_ptr<shared_var_base>(value.clone()));

        auto result = variables.insert(entry);

        if (result.second)
        {
            write_set.insert(var);
        }
        else
        {
            result.first->second = entry.second;
            write_set.insert(var);
        }
    }

    void then(future_type* fut, ASTM_FUNCTION<void(transaction*)> F)
    {
        async_list.emplace_back(fut, std::move(F));
    }
};

template <typename T>
shared_var<T>::local_var::operator T const& () const
{
    return dynamic_cast<shared_var const*>(&trans_->read(var_))->read();
}

template <typename T>
T shared_var<T>::local_var::get() const
{
    return dynamic_cast<shared_var const*>(&trans_->read(var_))->read();
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
    assert(var_ != nullptr);

    auto* typed = dynamic_cast<shared_var*>(var_);
    assert(typed != nullptr);

    trans_->then(&typed->queue, ASTM_FUNCTION<void(transaction*)>(f));
}

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_ != nullptr);
    trans_->then(&fut_, ASTM_FUNCTION<void(transaction*)>(f));
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
