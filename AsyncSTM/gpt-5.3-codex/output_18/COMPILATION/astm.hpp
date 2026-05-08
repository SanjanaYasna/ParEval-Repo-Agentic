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

#include <list>
#include <set>
#include <map>
#include <vector>
#include <memory>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <mutex>
#include <functional>
#include <unordered_map>
#include <utility>

namespace astm
{

struct transaction;

namespace detail
{
    // Internal Legion task used to execute queued async callbacks.
    static constexpr Legion::TaskID ASTM_ASYNC_TRAMPOLINE_TASK_ID = 100001;

    struct async_payload
    {
        std::uint64_t op_id;
        std::uintptr_t trans_ptr;
    };

    inline std::unordered_map<std::uint64_t, std::function<void(transaction*)>>& callback_table()
    {
        static std::unordered_map<std::uint64_t, std::function<void(transaction*)>> table;
        return table;
    }

    inline std::mutex& callback_table_mtx()
    {
        static std::mutex mtx;
        return mtx;
    }

    inline std::atomic<std::uint64_t>& next_callback_id()
    {
        static std::atomic<std::uint64_t> id{1};
        return id;
    }

    inline void async_trampoline_task(
        Legion::Task const* task,
        std::vector<Legion::PhysicalRegion> const&,
        Legion::Context,
        Legion::Runtime*)
    {
        assert(task != nullptr);
        assert(task->args != nullptr);
        assert(task->arglen == sizeof(async_payload));

        auto const* payload = static_cast<async_payload const*>(task->args);

        std::function<void(transaction*)> fn;
        {
            std::lock_guard<std::mutex> lk(callback_table_mtx());
            auto it = callback_table().find(payload->op_id);
            if (it != callback_table().end())
            {
                fn = std::move(it->second);
                callback_table().erase(it);
            }
        }

        if (fn)
            fn(reinterpret_cast<transaction*>(payload->trans_ptr));
    }

    struct preregister_async_task
    {
        preregister_async_task()
        {
            Legion::TaskVariantRegistrar registrar(
                ASTM_ASYNC_TRAMPOLINE_TASK_ID,
                "astm_async_trampoline_task");
            registrar.add_constraint(
                Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
            Legion::Runtime::preregister_task_variant<async_trampoline_task>(
                registrar, "astm_async_trampoline_task");
        }
    };

    // Header-only, one global registration point.
    inline preregister_async_task preregister_async_task_instance{};

    inline Legion::Future launch_async(
        std::function<void(transaction*)> fn,
        transaction* trans,
        Legion::Future const* dependency)
    {
        std::uint64_t const id = next_callback_id().fetch_add(1, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lk(callback_table_mtx());
            callback_table().emplace(id, std::move(fn));
        }

        async_payload payload{id, reinterpret_cast<std::uintptr_t>(trans)};
        Legion::TaskLauncher launcher(
            ASTM_ASYNC_TRAMPOLINE_TASK_ID,
            Legion::TaskArgument(&payload, sizeof(payload)));

        if (dependency != nullptr && dependency->exists())
            launcher.add_future(*dependency);

        Legion::Runtime* runtime = Legion::Runtime::get_runtime();
        Legion::Context ctx = Legion::Runtime::get_context();

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
            fut_.wait();
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

    ASTM_LOCK lock() const override
    {
        return ASTM_LOCK(mtx_);
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
        std::pair<
            Legion::Future*,
            std::function<void(transaction*)>>> async_list;

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
            var->write(*(it->second));
        }

        for (auto& op : async_list)
        {
            if (op.first == nullptr)
            {
                (void)detail::launch_async(std::move(op.second), this, nullptr);
            }
            else
            {
                Legion::Future dep = *(op.first);
                *(op.first) = detail::launch_async(
                    std::move(op.second), this, dep.exists() ? &dep : nullptr);
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
            return *(result.first->second);
        }

        return *(result.first->second);
    }

    void write(shared_var_base* var, shared_var_base const& value)
    {
        assert(var != nullptr);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base>> entry(
            var, std::shared_ptr<shared_var_base>(value.clone()));

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

    void then(Legion::Future* fut, std::function<void(transaction*)> F)
    {
        async_list.emplace_back(fut, std::move(F));
    }
};

template <typename T>
shared_var<T>::local_var::operator T const&() const
{
    auto const* p = dynamic_cast<shared_var<T> const*>(&trans_->read(var_));
    assert(p != nullptr);
    return p->read();
}

template <typename T>
T shared_var<T>::local_var::get() const
{
    auto const* p = dynamic_cast<shared_var<T> const*>(&trans_->read(var_));
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
    shared_var<T> tmp(rhs);
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
