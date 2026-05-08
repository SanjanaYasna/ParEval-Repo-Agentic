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

#include <cassert>
#include <atomic>
#include <cstring>
#include <future>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <functional>

namespace astm
{

struct shared_var_base;
struct transaction;
struct transaction_future;

namespace detail
{
    constexpr Legion::TaskID ASTM_INTERNAL_CONTINUATION_TASK_ID = 0x7FFF0101;

    inline std::shared_future<void> make_ready_shared_future()
    {
        std::promise<void> p;
        p.set_value();
        return p.get_future().share();
    }

    struct continuation_registry
    {
        std::mutex mtx;
        std::unordered_map<uint64_t, std::function<void()>> funcs;
        std::atomic<uint64_t> next_id{1};
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
        uint64_t id = 0;
        if (task != nullptr && task->args != nullptr && task->arglen == sizeof(uint64_t))
            std::memcpy(&id, task->args, sizeof(uint64_t));

        std::function<void()> fn;
        {
            std::lock_guard<std::mutex> lk(registry().mtx);
            auto it = registry().funcs.find(id);
            if (it != registry().funcs.end())
            {
                fn = std::move(it->second);
                registry().funcs.erase(it);
            }
        }

        if (fn) fn();
    }

    inline void preregister_internal_tasks()
    {
        static bool once = []() {
            Legion::TaskVariantRegistrar registrar(
                ASTM_INTERNAL_CONTINUATION_TASK_ID,
                "astm_internal_continuation");
            registrar.add_constraint(
                Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
            Legion::Runtime::preregister_task_variant<continuation_task>(
                registrar, "astm_internal_continuation");
            return true;
        }();
        (void)once;
    }

    inline Legion::Future launch_legion_continuation(
        Legion::Runtime* runtime,
        Legion::Context ctx,
        std::function<void()> fn,
        const Legion::Future* dep = nullptr)
    {
        preregister_internal_tasks();

        uint64_t id = registry().next_id.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(registry().mtx);
            registry().funcs.emplace(id, std::move(fn));
        }

        Legion::TaskLauncher launcher(
            ASTM_INTERNAL_CONTINUATION_TASK_ID,
            Legion::TaskArgument(&id, sizeof(uint64_t)));

        if (dep != nullptr)
            launcher.add_future(*dep);

        return runtime->execute_task(ctx, launcher);
    }
} // namespace detail

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
  private:
    transaction* trans_;
    std::shared_future<void> std_chain_;
    std::vector<Legion::Future> legion_chain_;

  public:
    explicit transaction_future(transaction* trans)
      : trans_(trans)
      , std_chain_(detail::make_ready_shared_future())
    {}

    explicit transaction_future(transaction& trans)
      : trans_(&trans)
      , std_chain_(detail::make_ready_shared_future())
    {}

    template <typename F>
    void then(F f);

    void get();

    friend struct transaction;
};

template <typename T>
struct shared_var : shared_var_base
{
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
    mutable std::mutex mtx_;

  public:
    shared_var()
      : data_()
      , mtx_()
    {}

    explicit shared_var(T const& t)
      : data_(t)
      , mtx_()
    {}

    explicit shared_var(T&& t)
      : data_(std::move(t))
      , mtx_()
    {}

    shared_var(shared_var const& rhs)
      : data_(rhs.data_)
      , mtx_()
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

    std::unique_lock<std::mutex> lock() const override
    {
        return std::unique_lock<std::mutex>(mtx_);
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
    std::list<
        std::pair<
            shared_var_base*,
            std::shared_ptr<shared_var_base>
        >
    > read_list;

    std::set<shared_var_base*> write_set;

    std::list<
        std::pair<
            transaction_future*,
            std::function<void(transaction*)>
        >
    > async_list;

    std::map<
        shared_var_base*,
        std::shared_ptr<shared_var_base>
    > variables;

    Legion::Runtime* runtime_;
    Legion::Context ctx_;

    transaction()
      : read_list()
      , write_set()
      , async_list()
      , variables()
      , runtime_(nullptr)
      , ctx_()
    {}

    transaction(Legion::Runtime* runtime, Legion::Context ctx)
      : read_list()
      , write_set()
      , async_list()
      , variables()
      , runtime_(runtime)
      , ctx_(ctx)
    {
        detail::preregister_internal_tasks();
    }

    void bind_legion_context(Legion::Runtime* runtime, Legion::Context ctx)
    {
        runtime_ = runtime;
        ctx_ = ctx;
        detail::preregister_internal_tasks();
    }

    bool has_legion_context() const
    {
        return runtime_ != nullptr;
    }

    void clear()
    {
        read_list.clear();
        write_set.clear();
        async_list.clear();
        variables.clear();
    }

    bool commit_transaction()
    {
        std::list<std::unique_lock<std::mutex>> locks;

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
            if (has_legion_context())
            {
                const Legion::Future* dep_ptr = nullptr;
                Legion::Future dep;

                if (op.first != nullptr && !op.first->legion_chain_.empty())
                {
                    dep = op.first->legion_chain_.back();
                    dep_ptr = &dep;
                }

                Legion::Future launched = detail::launch_legion_continuation(
                    runtime_, ctx_, std::bind(op.second, this), dep_ptr);

                if (op.first != nullptr)
                    op.first->legion_chain_.push_back(launched);
            }
            else
            {
                if (op.first == nullptr)
                {
                    std::thread([fn = op.second, this]() { fn(this); }).detach();
                }
                else
                {
                    auto prev = op.first->std_chain_;
                    auto next = std::async(
                        std::launch::async,
                        [prev, fn = op.second, this]() mutable {
                            prev.wait();
                            fn(this);
                        });
                    op.first->std_chain_ = next.share();
                }
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

    void then(transaction_future* fut, std::function<void(transaction*)> F)
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
    trans_->then(nullptr, std::function<void(transaction*)>(std::move(f)));
}

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_ != nullptr);
    trans_->then(this, std::function<void(transaction*)>(std::move(f)));
}

inline void transaction_future::get()
{
    for (auto& f : legion_chain_)
        f.get_void_result();

    std_chain_.wait();
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
