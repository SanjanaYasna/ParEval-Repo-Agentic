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
#include <cstring>
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

using mutex_type = std::mutex;
using lock_type = std::unique_lock<mutex_type>;

namespace detail
{
using continuation_id_t = std::uint64_t;

#ifndef ASTM_LEGION_CONTINUATION_TASK_ID
#define ASTM_LEGION_CONTINUATION_TASK_ID 0x5A57A001
#endif

inline constexpr Legion::TaskID continuation_task_id =
    static_cast<Legion::TaskID>(ASTM_LEGION_CONTINUATION_TASK_ID);

struct continuation_registry
{
    std::mutex mtx;
    std::unordered_map<continuation_id_t, std::function<void()>> callbacks;
    std::atomic<continuation_id_t> next_id{1};

    continuation_id_t emplace(std::function<void()> fn)
    {
        std::lock_guard<std::mutex> g(mtx);
        continuation_id_t id = next_id++;
        callbacks.emplace(id, std::move(fn));
        return id;
    }

    std::function<void()> take(continuation_id_t id)
    {
        std::lock_guard<std::mutex> g(mtx);
        auto it = callbacks.find(id);
        if (it == callbacks.end())
            return {};
        std::function<void()> fn = std::move(it->second);
        callbacks.erase(it);
        return fn;
    }
};

inline continuation_registry& registry()
{
    static continuation_registry r;
    return r;
}

inline void continuation_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>&,
    Legion::Context,
    Legion::Runtime*)
{
    if (task == nullptr || task->args == nullptr || task->arglen < int(sizeof(continuation_id_t)))
        return;

    continuation_id_t id = 0;
    std::memcpy(&id, task->args, sizeof(id));

    std::function<void()> fn = registry().take(id);
    if (fn)
        fn();
}

inline bool preregistered_continuation_task = []() {
    Legion::TaskVariantRegistrar registrar(continuation_task_id, "astm_continuation_task");
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    Legion::Runtime::preregister_task_variant<continuation_task>(
        registrar, "astm_continuation_task");
    return true;
}();

inline void ensure_registration()
{
    (void)preregistered_continuation_task;
}

inline Legion::Future launch_continuation(std::function<void()> fn, const Legion::Future* dep = nullptr)
{
    ensure_registration();

    Legion::Runtime* runtime = Legion::Runtime::get_runtime();
    Legion::Context context = Legion::Runtime::get_context();

    // Outside Legion task context: execute inline (best-effort fallback).
    if (runtime == nullptr || context == nullptr)
    {
        fn();
        return Legion::Future();
    }

    continuation_id_t id = registry().emplace(std::move(fn));
    Legion::TaskLauncher launcher(
        continuation_task_id, Legion::TaskArgument(&id, sizeof(id)));

    if (dep != nullptr && dep->exists())
        launcher.add_future(*dep);

    return runtime->execute_task(context, launcher);
}

} // namespace detail

class future_void
{
  public:
    future_void() : has_future_(false), fut_() {}
    explicit future_void(const Legion::Future& fut)
      : has_future_(fut.exists()), fut_(fut)
    {}

    void get()
    {
        if (has_future_ && fut_.exists())
            fut_.get_void_result();
    }

    template <typename F>
    future_void then(F&& f)
    {
        std::function<void()> fn(std::forward<F>(f));
        Legion::Future next =
            detail::launch_continuation(std::move(fn), (has_future_ && fut_.exists()) ? &fut_ : nullptr);
        return future_void(next);
    }

  private:
    bool has_future_;
    Legion::Future fut_;
};

struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;
    virtual void write(shared_var_base const&) = 0;
    virtual lock_type lock() const = 0;
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
      : trans_(trans), fut_()
    {}

    explicit transaction_future(transaction& trans)
      : trans_(&trans), fut_()
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
    mutable mutex_type mtx_;

  public:
    future_type queue;

    shared_var() : data_(), mtx_(), queue() {}
    explicit shared_var(T const& t) : data_(t), mtx_(), queue() {}
    explicit shared_var(T&& t) : data_(std::move(t)), mtx_(), queue() {}
    shared_var(shared_var const& rhs) : data_(rhs.data_), mtx_(), queue() {}

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
    using future_type = future_void;

    std::list<
        std::pair<shared_var_base*, std::shared_ptr<shared_var_base>>
    > read_list;

    std::set<shared_var_base*> write_set;

    std::list<
        std::pair<future_type*, std::function<void(transaction*)>>
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
        std::list<lock_type> locks;

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
                detail::launch_continuation(std::bind(op.second, this), nullptr);
            }
            else
            {
                *op.first = op.first->then(std::bind(op.second, this));
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

    void then(future_type* fut, std::function<void(transaction*)> F)
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
void shared_var<T>::local_var::then(F f)
{
    assert(trans_ != nullptr);
    auto* concrete = dynamic_cast<shared_var<T>*>(var_);
    assert(concrete != nullptr);
    trans_->then(&concrete->queue, std::function<void(transaction*)>(f));
}

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_ != nullptr);
    trans_->then(&fut_, std::function<void(transaction*)>(f));
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
