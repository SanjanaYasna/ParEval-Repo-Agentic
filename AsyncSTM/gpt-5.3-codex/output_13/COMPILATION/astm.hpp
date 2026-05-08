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
#include <cstdint>
#include <future>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

namespace astm
{

struct transaction;

// Bind Legion runtime/context from your top-level Legion task before using
// transaction::then/transaction_future::then:
//
//   astm::bind_legion_context(runtime, ctx);
inline void bind_legion_context(Legion::Runtime* runtime, Legion::Context ctx);
inline void unbind_legion_context();

class future_void
{
public:
    enum class backend_kind
    {
        ready,
        legion,
        std_async
    };

    future_void()
      : backend_(backend_kind::ready)
      , legion_future_()
      , std_future_()
    {}

    void set_ready()
    {
        backend_ = backend_kind::ready;
        legion_future_ = Legion::Future();
        std_future_ = std::shared_future<void>();
    }

    void set_legion(Legion::Future f)
    {
        backend_ = backend_kind::legion;
        legion_future_ = std::move(f);
        std_future_ = std::shared_future<void>();
    }

    void set_std(std::shared_future<void> f)
    {
        backend_ = backend_kind::std_async;
        std_future_ = std::move(f);
        legion_future_ = Legion::Future();
    }

    backend_kind kind() const
    {
        return backend_;
    }

    bool is_ready() const
    {
        return backend_ == backend_kind::ready;
    }

    bool is_legion() const
    {
        return backend_ == backend_kind::legion;
    }

    bool is_std() const
    {
        return backend_ == backend_kind::std_async;
    }

    Legion::Future const& legion_future() const
    {
        return legion_future_;
    }

    std::shared_future<void> const& std_future() const
    {
        return std_future_;
    }

    void get() const
    {
        switch (backend_)
        {
        case backend_kind::ready:
            return;

        case backend_kind::legion:
            legion_future_.get_void_result();
            return;

        case backend_kind::std_async:
            if (std_future_.valid())
                std_future_.wait();
            return;
        }
    }

private:
    backend_kind backend_;
    Legion::Future legion_future_;
    std::shared_future<void> std_future_;
};

namespace detail
{
    inline constexpr Legion::TaskID ASTM_LEGION_ASYNC_TASK_ID = 0xA57A0001;

    struct legion_binding
    {
        Legion::Runtime* runtime = nullptr;
        Legion::Context context = Legion::Context();
    };

    inline legion_binding& get_binding()
    {
        static thread_local legion_binding binding;
        return binding;
    }

    struct async_closure_base
    {
        virtual ~async_closure_base() = default;
        virtual void run() = 0;
    };

    struct async_closure final : async_closure_base
    {
        std::function<void(transaction*)> fn;
        transaction* trans;

        async_closure(std::function<void(transaction*)> f, transaction* t)
          : fn(std::move(f))
          , trans(t)
        {}

        void run() override
        {
            fn(trans);
        }
    };

    inline void astm_legion_async_task(
        Legion::Task const* task,
        std::vector<Legion::PhysicalRegion> const&,
        Legion::Context,
        Legion::Runtime*)
    {
        assert(task != nullptr);
        assert(task->arglen == static_cast<int>(sizeof(std::uint64_t)));

        std::uint64_t raw = *reinterpret_cast<std::uint64_t const*>(task->args);
        auto* closure = reinterpret_cast<async_closure_base*>(raw);

        assert(closure != nullptr);
        closure->run();
        delete closure;
    }

    inline void preregister_astm_legion_tasks()
    {
        static bool registered = false;
        if (registered)
            return;

        Legion::TaskVariantRegistrar registrar(
            ASTM_LEGION_ASYNC_TASK_ID, "astm_legion_async_task");
        registrar.add_constraint(
            Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        registrar.set_leaf();

        Legion::Runtime::preregister_task_variant<astm_legion_async_task>(
            registrar, "astm_legion_async_task");

        registered = true;
    }

    struct preregister_guard
    {
        preregister_guard()
        {
            preregister_astm_legion_tasks();
        }
    };

    inline preregister_guard preregister_guard_instance{};

    inline bool has_legion_context()
    {
        auto const& b = get_binding();
        return (b.runtime != nullptr) && (b.context != Legion::Context());
    }

    inline future_void launch_async(
        std::function<void(transaction*)> fn,
        transaction* trans,
        future_void const* dependency)
    {
        future_void out;

        if (has_legion_context())
        {
            // If dependency is non-Legion, satisfy it before launching.
            if (dependency != nullptr && !dependency->is_legion())
                dependency->get();

            auto* closure = new async_closure(std::move(fn), trans);
            std::uint64_t raw = reinterpret_cast<std::uint64_t>(closure);

            Legion::TaskLauncher launcher(
                ASTM_LEGION_ASYNC_TASK_ID,
                Legion::TaskArgument(&raw, sizeof(raw)));

            if (dependency != nullptr && dependency->is_legion())
                launcher.add_future(dependency->legion_future());

            auto& b = get_binding();
            Legion::Future f = b.runtime->execute_task(b.context, launcher);

            out.set_legion(std::move(f));
            return out;
        }

        future_void dep = (dependency != nullptr) ? *dependency : future_void();
        auto sf = std::async(std::launch::async,
            [dep, fn = std::move(fn), trans]() mutable {
                dep.get();
                fn(trans);
            }).share();

        out.set_std(std::move(sf));
        return out;
    }

} // namespace detail

inline void bind_legion_context(Legion::Runtime* runtime, Legion::Context ctx)
{
    auto& b = detail::get_binding();
    b.runtime = runtime;
    b.context = ctx;
}

inline void unbind_legion_context()
{
    auto& b = detail::get_binding();
    b.runtime = nullptr;
    b.context = Legion::Context();
}

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
    mutable std::mutex mtx_;

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
            future_void*,
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
        // 1) Lock all accessed vars in deterministic order (map order).
        std::list<std::unique_lock<std::mutex>> locks;
        for (auto const& var : variables)
        {
            assert(var.first != nullptr);
            locks.push_back(var.first->lock());
        }

        // 2) Validate read set.
        for (auto const& entry : read_list)
        {
            assert(entry.first != nullptr);
            if (!((*entry.first) == (*entry.second)))
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

        // 4) Launch async operations.
        for (auto& op : async_list)
        {
            if (op.first == nullptr)
            {
                (void)detail::launch_async(op.second, this, nullptr);
            }
            else
            {
                *op.first = detail::launch_async(op.second, this, op.first);
            }
        }

        // 5) Locks released by RAII.
        return true;
    }

    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != nullptr);

        auto result = variables.insert(std::make_pair(var, std::shared_ptr<shared_var_base>()));
        if (result.second)
        {
            // First access to var in this transaction: clone current value.
            result.first->second.reset(var->clone());

            // Store an immutable snapshot for validation.
            read_list.push_back(std::make_pair(
                var, std::shared_ptr<shared_var_base>(result.first->second->clone())));
        }

        return *(result.first->second);
    }

    void write(shared_var_base* var, shared_var_base const& value)
    {
        assert(var != nullptr);

        auto entry = std::make_pair(
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

    void then(future_void* fut, std::function<void(transaction*)> F)
    {
        async_list.push_back(std::make_pair(fut, std::move(F)));
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
    auto* concrete = dynamic_cast<shared_var<T>*>(var_);
    assert(concrete != nullptr);

    std::function<void(transaction*)> fn = f;
    trans_->then(&concrete->queue, std::move(fn));
}

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_ != nullptr);
    std::function<void(transaction*)> fn = f;
    trans_->then(&fut_, std::move(fn));
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94