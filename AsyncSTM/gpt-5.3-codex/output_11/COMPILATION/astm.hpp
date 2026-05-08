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

struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;
    virtual void write(shared_var_base const&) = 0;
    virtual ASTM_LOCK lock() const = 0;
    virtual bool operator==(shared_var_base const&) const = 0;
};

struct transaction;

//------------------------------------------------------------------------------
// Legion runtime/context binding for ASTM.
// Call astm::bind_legion_context(runtime, ctx) at the beginning of each Legion
// task that uses ASTM transactions.
//------------------------------------------------------------------------------
namespace detail
{
    struct legion_binding_state
    {
        Legion::Runtime* runtime{nullptr};
        Legion::Context context{};
        bool bound{false};
    };

    inline thread_local legion_binding_state tls_binding{};

    inline void bind_legion_context(Legion::Runtime* runtime, Legion::Context context)
    {
        tls_binding.runtime = runtime;
        tls_binding.context = context;
        tls_binding.bound = (runtime != nullptr);
    }

    inline void unbind_legion_context()
    {
        tls_binding.runtime = nullptr;
        tls_binding.context = Legion::Context{};
        tls_binding.bound = false;
    }

    inline bool has_legion_context()
    {
        return tls_binding.bound;
    }

    inline Legion::Runtime* get_runtime()
    {
        return tls_binding.runtime;
    }

    inline Legion::Context get_context()
    {
        return tls_binding.context;
    }

    using async_callback_type = ASTM_FUNCTION<void(transaction*)>;

    inline std::mutex& callback_mutex()
    {
        static std::mutex m;
        return m;
    }

    inline std::unordered_map<uint64_t, async_callback_type>& callback_table()
    {
        static std::unordered_map<uint64_t, async_callback_type> table;
        return table;
    }

    inline std::atomic<uint64_t>& next_callback_id()
    {
        static std::atomic<uint64_t> id{1};
        return id;
    }

    inline uint64_t store_callback(async_callback_type cb)
    {
        const uint64_t id = next_callback_id().fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(callback_mutex());
        callback_table().emplace(id, std::move(cb));
        return id;
    }

    inline async_callback_type take_callback(uint64_t id)
    {
        std::lock_guard<std::mutex> lk(callback_mutex());
        auto it = callback_table().find(id);
        if (it == callback_table().end())
            return async_callback_type{};
        async_callback_type fn = std::move(it->second);
        callback_table().erase(it);
        return fn;
    }

    struct async_task_payload
    {
        uint64_t callback_id;
        uintptr_t transaction_ptr;
    };

    inline constexpr Legion::TaskID ASTM_ASYNC_TASK_ID = 1000001;

    inline void astm_async_task(
        const Legion::Task* task,
        const std::vector<Legion::PhysicalRegion>&,
        Legion::Context,
        Legion::Runtime*)
    {
        if (task == nullptr || task->args == nullptr ||
            task->arglen != static_cast<int>(sizeof(async_task_payload)))
            return;

        const auto* payload = static_cast<const async_task_payload*>(task->args);
        auto cb = take_callback(payload->callback_id);
        if (cb)
        {
            transaction* trans = reinterpret_cast<transaction*>(payload->transaction_ptr);
            cb(trans);
        }
    }

    inline void preregister_astm_tasks()
    {
        static bool registered = false;
        if (registered) return;

        Legion::TaskVariantRegistrar registrar(ASTM_ASYNC_TASK_ID, "astm_async_task");
        registrar.add_constraint(
            Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));

        Legion::Runtime::preregister_task_variant<astm_async_task>(
            registrar, "astm_async_task");

        registered = true;
    }

    struct astm_task_registration_once
    {
        astm_task_registration_once()
        {
            preregister_astm_tasks();
        }
    };

    inline astm_task_registration_once astm_task_registration_guard{};
} // namespace detail

inline void bind_legion_context(Legion::Runtime* runtime, Legion::Context context)
{
    detail::bind_legion_context(runtime, context);
}

inline void unbind_legion_context()
{
    detail::unbind_legion_context();
}

//------------------------------------------------------------------------------

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
    future_type queue;

    shared_var() : data_(), mtx_(), queue() {}
    shared_var(T const& t) : data_(t), mtx_(), queue() {}
    shared_var(T&& t) : data_(std::move(t)), mtx_(), queue() {}
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
    using async_function_type = ASTM_FUNCTION<void(transaction*)>;

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
            async_function_type
        >
    > async_list;

    std::map<
        shared_var_base*,
        std::shared_ptr<shared_var_base>
    > variables;

    Legion::Runtime* runtime_;
    Legion::Context context_;
    bool has_legion_context_;

    transaction()
      : read_list()
      , write_set()
      , async_list()
      , variables()
      , runtime_(detail::get_runtime())
      , context_(detail::get_context())
      , has_legion_context_(detail::has_legion_context())
    {}

    void clear()
    {
        read_list.clear();
        write_set.clear();
        async_list.clear();
        variables.clear();
    }

    bool commit_transaction()
    {
        // 1.) Obtain exclusive access to all tracked variables.
        std::list<ASTM_LOCK> locks;
        for (auto const& var : variables)
        {
            assert(var.first != nullptr);
            locks.push_back(var.first->lock());
        }

        // 2.) Validate reads.
        for (auto const& var : read_list)
        {
            assert(var.first != nullptr);
            if (!((*var.first) == (*var.second)))
            {
                clear();
                return false;
            }
        }

        // 3.) Apply writes.
        for (shared_var_base* var : write_set)
        {
            assert(var != nullptr);
            auto it = variables.find(var);
            assert(it != variables.end());
            var->write(*it->second);
        }

        // Refresh context if it was bound after transaction construction.
        if (!has_legion_context_ && detail::has_legion_context())
        {
            runtime_ = detail::get_runtime();
            context_ = detail::get_context();
            has_legion_context_ = true;
        }

        // 4.) Execute async continuations.
        for (auto& op : async_list)
        {
            if (has_legion_context_ && runtime_ != nullptr)
            {
                detail::async_task_payload payload;
                payload.callback_id = detail::store_callback(op.second);
                payload.transaction_ptr = reinterpret_cast<uintptr_t>(this);

                Legion::TaskLauncher launcher(
                    detail::ASTM_ASYNC_TASK_ID,
                    Legion::TaskArgument(&payload, sizeof(payload)));

                if (op.first != nullptr && op.first->exists())
                    launcher.add_future(*op.first);

                Legion::Future launched = runtime_->execute_task(context_, launcher);

                if (op.first != nullptr)
                    *op.first = launched;
            }
            else
            {
                // Fallback (outside Legion task context): execute inline.
                if (op.first != nullptr && op.first->exists())
                    op.first->get_void_result();

                op.second(this);

                if (op.first != nullptr)
                    *op.first = Legion::Future{};
            }
        }

        // 5.) Locks released via RAII.
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
        else
        {
            return *result.first->second;
        }
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

    // If fut is nullptr, fire-and-forget semantics are used.
    void then(future_type* fut, async_function_type F)
    {
        async_list.emplace_back(fut, std::move(F));
    }
};

template <typename T>
shared_var<T>::local_var::operator T const& () const
{
    return dynamic_cast<shared_var<T> const*>(&trans_->read(var_))->read();
}

template <typename T>
T shared_var<T>::local_var::get() const
{
    return dynamic_cast<shared_var<T> const*>(&trans_->read(var_))->read();
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

    auto* typed_var = dynamic_cast<shared_var<T>*>(var_);
    assert(typed_var != nullptr);

    trans_->then(
        &typed_var->queue,
        typename transaction::async_function_type(f));
}

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_ != nullptr);
    trans_->then(&fut_, typename transaction::async_function_type(f));
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
