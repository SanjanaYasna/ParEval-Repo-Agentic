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

struct transaction;

// ----------------------------------------------------------------------------
// Legion context binding helpers.
// Call bind_legion_context(runtime, ctx) at the beginning of a Legion task
// before using ASTM transactional async continuations.
// ----------------------------------------------------------------------------
namespace detail
{
    constexpr Legion::TaskID ASTM_INTERNAL_CONTINUATION_TASK_ID = 0xAC710001;

    struct continuation_payload
    {
        std::uint64_t callback_id;
        std::uintptr_t transaction_ptr;
    };

    inline thread_local Legion::Runtime* tls_runtime = nullptr;
    inline thread_local Legion::Context tls_context = Legion::Context();
    inline thread_local bool tls_bound = false;

    inline std::mutex& callback_mutex()
    {
        static std::mutex m;
        return m;
    }

    inline std::unordered_map<std::uint64_t, std::function<void(transaction*)>>& callback_table()
    {
        static std::unordered_map<std::uint64_t, std::function<void(transaction*)>> table;
        return table;
    }

    inline std::atomic<std::uint64_t>& next_callback_id()
    {
        static std::atomic<std::uint64_t> id{1};
        return id;
    }

    inline void bind_context(Legion::Runtime* runtime, Legion::Context context)
    {
        tls_runtime = runtime;
        tls_context = context;
        tls_bound = (runtime != nullptr);
    }

    inline void unbind_context()
    {
        tls_runtime = nullptr;
        tls_context = Legion::Context();
        tls_bound = false;
    }

    inline bool has_bound_context()
    {
        return tls_bound;
    }

    inline void continuation_task(
        Legion::Task const* task,
        std::vector<Legion::PhysicalRegion> const&,
        Legion::Context,
        Legion::Runtime*)
    {
        assert(task != nullptr);
        assert(task->arglen == sizeof(continuation_payload));

        continuation_payload payload =
            *reinterpret_cast<continuation_payload const*>(task->args);

        std::function<void(transaction*)> fn;
        {
            std::lock_guard<std::mutex> l(callback_mutex());
            auto& table = callback_table();
            auto it = table.find(payload.callback_id);
            if (it != table.end())
            {
                fn = std::move(it->second);
                table.erase(it);
            }
        }

        if (fn)
            fn(reinterpret_cast<transaction*>(payload.transaction_ptr));
    }

    inline void preregister_internal_tasks()
    {
        static std::once_flag once;
        std::call_once(once, [] {
            Legion::TaskVariantRegistrar registrar(
                ASTM_INTERNAL_CONTINUATION_TASK_ID,
                "astm_internal_continuation_task");
            registrar.add_constraint(
                Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
            registrar.set_leaf();

            Legion::Runtime::preregister_task_variant<continuation_task>(
                registrar, "astm_internal_continuation_task");
        });
    }

    inline Legion::Future launch_continuation(
        std::function<void(transaction*)> fn,
        transaction* trans,
        Legion::Future const& dependency = Legion::Future())
    {
        assert(has_bound_context() &&
               "ASTM Legion context not bound. Call astm::bind_legion_context "
               "inside the current Legion task first.");
        assert(trans != nullptr);

        preregister_internal_tasks();

        std::uint64_t id = next_callback_id().fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> l(callback_mutex());
            callback_table().emplace(id, std::move(fn));
        }

        continuation_payload payload{id, reinterpret_cast<std::uintptr_t>(trans)};
        Legion::TaskLauncher launcher(
            ASTM_INTERNAL_CONTINUATION_TASK_ID,
            Legion::TaskArgument(&payload, sizeof(payload)));

        if (dependency.exists())
            launcher.add_future(dependency);

        return tls_runtime->execute_task(tls_context, launcher);
    }
} // namespace detail

inline void bind_legion_context(Legion::Runtime* runtime, Legion::Context context)
{
    detail::bind_context(runtime, context);
    detail::preregister_internal_tasks();
}

inline void unbind_legion_context()
{
    detail::unbind_context();
}

struct legion_context_scope
{
    legion_context_scope(Legion::Runtime* runtime, Legion::Context context)
    {
        bind_legion_context(runtime, context);
    }

    ~legion_context_scope()
    {
        unbind_legion_context();
    }
};

// Force preregistration in any TU that includes this header before Runtime::start.
inline const bool astm_legion_preregistered = [] {
    detail::preregister_internal_tasks();
    return true;
}();

// ----------------------------------------------------------------------------
// STM core
// ----------------------------------------------------------------------------
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
        auto const* typed = dynamic_cast<shared_var const*>(&rhs);
        assert(typed != nullptr);
        data_ = typed->read();
    }

    ASTM_LOCK lock() const override
    {
        return ASTM_LOCK(mtx_);
    }

    bool operator==(shared_var_base const& rhs) const override
    {
        auto const* typed = dynamic_cast<shared_var const*>(&rhs);
        assert(typed != nullptr);
        return data_ == typed->read();
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
            std::shared_ptr<shared_var_base>>> read_list;

    std::set<shared_var_base*> write_set;

    std::list<
        std::pair<
            Legion::Future*,
            std::function<void(transaction*)>>> async_list;

    std::map<
        shared_var_base*,
        std::shared_ptr<shared_var_base>> variables;

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

        // Legion-backed continuations (default mapper, no custom mapping).
        for (auto& op : async_list)
        {
            if (op.first == nullptr)
            {
                (void)detail::launch_continuation(op.second, this);
            }
            else
            {
                *op.first = detail::launch_continuation(op.second, this, *op.first);
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
    auto const* typed = dynamic_cast<shared_var const*>(&trans_->read(var_));
    assert(typed != nullptr);
    return typed->read();
}

template <typename T>
T shared_var<T>::local_var::get() const
{
    auto const* typed = dynamic_cast<shared_var const*>(&trans_->read(var_));
    assert(typed != nullptr);
    return typed->read();
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
    auto* typed = dynamic_cast<shared_var*>(var_);
    assert(typed != nullptr);
    trans_->then(&typed->queue, std::function<void(transaction*)>(f));
}

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_ != nullptr);
    trans_->then(&fut_, std::function<void(transaction*)>(f));
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
