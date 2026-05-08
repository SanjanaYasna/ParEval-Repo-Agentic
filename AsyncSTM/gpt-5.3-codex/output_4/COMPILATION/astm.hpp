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
#include <future>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace astm
{

struct transaction;

namespace detail
{
using mutex_type = std::mutex;
using lock_type  = std::unique_lock<mutex_type>;

//------------------------------------------------------------------------------
// Legion runtime/context binding
//------------------------------------------------------------------------------
//
// Call astm::bind_legion_context(runtime, ctx) once from your Legion top-level
// task before using ASTM transactions that schedule async continuations.
//
struct legion_binding
{
    mutex_type mtx;
    Legion::Runtime* runtime{nullptr};
    Legion::Context context{};
};

inline legion_binding& binding()
{
    static legion_binding b;
    return b;
}

inline void bind_legion_context(Legion::Runtime* runtime, Legion::Context ctx)
{
    std::lock_guard<mutex_type> lk(binding().mtx);
    binding().runtime = runtime;
    binding().context = ctx;
}

inline std::pair<Legion::Runtime*, Legion::Context> get_legion_context()
{
    std::lock_guard<mutex_type> lk(binding().mtx);
    return {binding().runtime, binding().context};
}

//------------------------------------------------------------------------------
// Continuation callback registry used by Legion task-based async continuations.
//------------------------------------------------------------------------------
struct continuation_task_args
{
    std::uint64_t callback_id;
    std::uintptr_t transaction_ptr;
};

inline std::unordered_map<std::uint64_t, std::function<void(transaction*)>>& callback_table()
{
    static std::unordered_map<std::uint64_t, std::function<void(transaction*)>> table;
    return table;
}

inline mutex_type& callback_table_mutex()
{
    static mutex_type m;
    return m;
}

inline std::atomic<std::uint64_t>& callback_counter()
{
    static std::atomic<std::uint64_t> c{1};
    return c;
}

inline std::uint64_t register_callback(std::function<void(transaction*)> fn)
{
    const std::uint64_t id = callback_counter().fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<mutex_type> lk(callback_table_mutex());
    callback_table().emplace(id, std::move(fn));
    return id;
}

inline std::function<void(transaction*)> take_callback(std::uint64_t id)
{
    std::lock_guard<mutex_type> lk(callback_table_mutex());
    auto it = callback_table().find(id);
    if (it == callback_table().end())
        return {};
    auto fn = std::move(it->second);
    callback_table().erase(it);
    return fn;
}

inline constexpr Legion::TaskID ASTM_LEGION_CONTINUATION_TASK_ID = 0xA57D0001;

inline void continuation_task(
    Legion::Task const* task,
    std::vector<Legion::PhysicalRegion> const&,
    Legion::Context,
    Legion::Runtime*)
{
    assert(task != nullptr);
    assert(task->arglen == sizeof(continuation_task_args));

    auto const* args = static_cast<continuation_task_args const*>(task->args);
    auto fn = take_callback(args->callback_id);
    if (fn)
    {
        auto* trans = reinterpret_cast<transaction*>(args->transaction_ptr);
        fn(trans);
    }
}

// Header-only preregistration.
struct continuation_task_registration
{
    continuation_task_registration()
    {
        Legion::TaskVariantRegistrar registrar(
            ASTM_LEGION_CONTINUATION_TASK_ID, "astm_continuation_task");
        registrar.add_constraint(
            Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<continuation_task>(
            registrar, "astm_continuation_task");
    }
};

inline continuation_task_registration continuation_registration_guard{};

//------------------------------------------------------------------------------
// Utility: normalize async callbacks to std::function<void(transaction*)>
//------------------------------------------------------------------------------
template <typename F>
inline std::function<void(transaction*)> make_async_action(F&& f)
{
    using FD = std::decay_t<F>;
    if constexpr (std::is_invocable_v<FD, transaction*>)
    {
        return [fn = FD(std::forward<F>(f))](transaction* t) mutable { fn(t); };
    }
    else if constexpr (std::is_invocable_v<FD>)
    {
        return [fn = FD(std::forward<F>(f))](transaction*) mutable { fn(); };
    }
    else
    {
        static_assert(
            std::is_invocable_v<FD, transaction*> || std::is_invocable_v<FD>,
            "Callback must be invocable as f(transaction*) or f().");
        return {};
    }
}

//------------------------------------------------------------------------------
// Async continuation chain
//------------------------------------------------------------------------------
class async_chain
{
public:
    async_chain() = default;

    void append(std::function<void(transaction*)> fn, transaction* trans)
    {
        auto [runtime, ctx] = get_legion_context();
        if (runtime != nullptr)
        {
            continuation_task_args args{
                register_callback(std::move(fn)),
                reinterpret_cast<std::uintptr_t>(trans)
            };

            std::lock_guard<mutex_type> lk(mtx_);
            Legion::TaskLauncher launcher(
                ASTM_LEGION_CONTINUATION_TASK_ID,
                Legion::TaskArgument(&args, sizeof(args)));

            if (legion_tail_.exists())
                launcher.add_future(legion_tail_);

            legion_tail_ = runtime->execute_task(ctx, launcher);
        }
        else
        {
            std::shared_future<void> prev;
            {
                std::lock_guard<mutex_type> lk(mtx_);
                prev = host_tail_;
                host_tail_ = std::async(
                                 std::launch::async,
                                 [prev, fn = std::move(fn), trans]() mutable {
                                     if (prev.valid()) prev.wait();
                                     fn(trans);
                                 })
                                 .share();
            }
        }
    }

    void wait() const
    {
        Legion::Future lf;
        std::shared_future<void> hf;
        {
            std::lock_guard<mutex_type> lk(mtx_);
            lf = legion_tail_;
            hf = host_tail_;
        }

        if (lf.exists())
            lf.get_void_result();
        if (hf.valid())
            hf.wait();
    }

private:
    mutable mutex_type mtx_;
    Legion::Future legion_tail_;
    std::shared_future<void> host_tail_;
};

inline void launch_detached(std::function<void(transaction*)> fn, transaction* trans)
{
    auto [runtime, ctx] = get_legion_context();
    if (runtime != nullptr)
    {
        continuation_task_args args{
            register_callback(std::move(fn)),
            reinterpret_cast<std::uintptr_t>(trans)
        };

        Legion::TaskLauncher launcher(
            ASTM_LEGION_CONTINUATION_TASK_ID,
            Legion::TaskArgument(&args, sizeof(args)));
        (void)runtime->execute_task(ctx, launcher); // fire-and-forget
    }
    else
    {
        std::thread([fn = std::move(fn), trans]() mutable { fn(trans); }).detach();
    }
}

} // namespace detail

// Public helper to bind Legion runtime/context into ASTM.
inline void bind_legion_context(Legion::Runtime* runtime, Legion::Context ctx)
{
    detail::bind_legion_context(runtime, ctx);
}

struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;
    virtual void write(shared_var_base const&) = 0;
    virtual detail::lock_type lock() const = 0;
    virtual bool operator==(shared_var_base const&) const = 0;
};

struct transaction_future
{
  private:
    transaction* trans_;
    std::shared_ptr<detail::async_chain> chain_;

  public:
    explicit transaction_future(transaction* trans)
      : trans_(trans)
      , chain_(std::make_shared<detail::async_chain>())
    {}

    explicit transaction_future(transaction& trans)
      : trans_(&trans)
      , chain_(std::make_shared<detail::async_chain>())
    {}

    template <typename F>
    void then(F&& f);

    void get()
    {
        chain_->wait();
    }

    detail::async_chain* chain_ptr() const
    {
        return chain_.get();
    }
};

template <typename T>
struct shared_var : shared_var_base
{
    using future_type = detail::async_chain;

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
    mutable detail::mutex_type mtx_;

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

    // No lock: caller controls transactional safety.
    T const& read() const
    {
        return data_;
    }

    // No lock: caller controls transactional safety.
    void write(T const& rhs)
    {
        data_ = rhs;
    }

    // No lock: caller controls transactional safety.
    void write(shared_var_base const& rhs) override
    {
        data_ = dynamic_cast<shared_var const*>(&rhs)->read();
    }

    detail::lock_type lock() const override
    {
        return detail::lock_type(mtx_);
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
    using async_action = std::function<void(transaction*)>;

    std::list<
        std::pair<
            shared_var_base*,
            std::shared_ptr<shared_var_base>
        >
    > read_list;

    std::set<shared_var_base*> write_set;

    std::list<
        std::pair<
            detail::async_chain*, // nullptr => fire-and-forget
            async_action
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
        // 1) Lock all variables in sorted order (map order).
        std::list<detail::lock_type> locks;
        for (auto const& var : variables)
        {
            assert(var.first != nullptr);
            locks.push_back(var.first->lock());
        }

        // 2) Validate reads.
        for (auto const& var : read_list)
        {
            assert(var.first != nullptr);
            if (!((*var.first) == (*var.second)))
            {
                clear();
                return false;
            }
        }

        // 3) Apply writes.
        for (shared_var_base* var : write_set)
        {
            assert(var != nullptr);
            auto it = variables.find(var);
            assert(it != variables.end());
            var->write(*it->second);
        }

        // 4) Launch async continuations.
        for (auto& op : async_list)
        {
            if (op.first == nullptr)
                detail::launch_detached(op.second, this);
            else
                op.first->append(op.second, this);
        }

        // 5) Locks released by RAII.
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

    void then(detail::async_chain* chain, async_action F)
    {
        async_list.emplace_back(chain, std::move(F));
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
void shared_var<T>::local_var::then(F&& f)
{
    assert(trans_ != nullptr);
    assert(var_ != nullptr);

    auto* concrete = dynamic_cast<shared_var<T>*>(var_);
    assert(concrete != nullptr);

    trans_->then(&concrete->queue, detail::make_async_action(std::forward<F>(f)));
}

template <typename F>
void transaction_future::then(F&& f)
{
    assert(trans_ != nullptr);
    trans_->then(chain_.get(), detail::make_async_action(std::forward<F>(f)));
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
