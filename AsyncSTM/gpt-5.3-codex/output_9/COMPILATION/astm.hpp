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
#include "legion.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <atomic>
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

struct transaction;

// Lightweight future handle used by transaction continuations and shared_var queues.
struct async_future
{
    enum class backend_kind { none, legion, std_async };

    backend_kind backend = backend_kind::none;
    Legion::Future legion_fut;
    std::shared_future<void> std_fut;

    bool valid() const { return backend != backend_kind::none; }
    bool is_legion() const { return backend == backend_kind::legion; }
    bool is_std() const { return backend == backend_kind::std_async; }

    void set_legion(Legion::Future f)
    {
        backend = backend_kind::legion;
        legion_fut = std::move(f);
    }

    void set_std(std::shared_future<void> f)
    {
        backend = backend_kind::std_async;
        std_fut = std::move(f);
    }

    void get()
    {
        if (backend == backend_kind::legion)
            legion_fut.get_void_result();
        else if (backend == backend_kind::std_async)
            std_fut.wait();
    }
};

namespace detail
{
    static constexpr Legion::TaskID ASTM_INTERNAL_ASYNC_TASK_ID = 0xA57B0001;

    struct legion_tls_state
    {
        Legion::Runtime* runtime = nullptr;
        Legion::Context context = Legion::Context();
        bool bound = false;
    };

    inline legion_tls_state& tls_state()
    {
        static thread_local legion_tls_state s;
        return s;
    }

    inline std::atomic<bool>& task_registered_flag()
    {
        static std::atomic<bool> reg{false};
        return reg;
    }

    struct async_task_payload
    {
        std::uint64_t callback_id;
        std::uintptr_t transaction_ptr;
    };

    struct callback_registry
    {
        static std::uint64_t insert(std::function<void(transaction*)> fn)
        {
            static std::atomic<std::uint64_t> next_id{1};
            static std::mutex mtx;
            static std::unordered_map<std::uint64_t, std::function<void(transaction*)>> table;

            const std::uint64_t id = next_id.fetch_add(1, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lk(mtx);
                table.emplace(id, std::move(fn));
            }
            return id;
        }

        static std::function<void(transaction*)> take(std::uint64_t id)
        {
            static std::mutex mtx;
            static std::unordered_map<std::uint64_t, std::function<void(transaction*)>> table_ref_hack;
            // Access the same static objects as in insert():
            // C++ does not allow direct shared local statics across functions,
            // so we use function-scope statics in both functions with identical
            // declarations guarded by the same ABI symbol in this translation unit.
            // To keep it robust in header-only form, re-route through lambdas below.

            auto& table = []() -> std::unordered_map<std::uint64_t, std::function<void(transaction*)>>& {
                static std::unordered_map<std::uint64_t, std::function<void(transaction*)>> t;
                return t;
            }();

            auto& mutex = []() -> std::mutex& {
                static std::mutex m;
                return m;
            }();

            (void)table_ref_hack;
            (void)mtx;

            std::function<void(transaction*)> out;
            std::lock_guard<std::mutex> lk(mutex);
            auto it = table.find(id);
            if (it != table.end())
            {
                out = std::move(it->second);
                table.erase(it);
            }
            return out;
        }
    };

    inline std::unordered_map<std::uint64_t, std::function<void(transaction*)>>& callback_table()
    {
        static std::unordered_map<std::uint64_t, std::function<void(transaction*)>> table;
        return table;
    }

    inline std::mutex& callback_table_mutex()
    {
        static std::mutex mtx;
        return mtx;
    }

    inline std::uint64_t store_callback(std::function<void(transaction*)> fn)
    {
        static std::atomic<std::uint64_t> next_id{1};
        const std::uint64_t id = next_id.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(callback_table_mutex());
            callback_table().emplace(id, std::move(fn));
        }
        return id;
    }

    inline std::function<void(transaction*)> pop_callback(std::uint64_t id)
    {
        std::function<void(transaction*)> fn;
        std::lock_guard<std::mutex> lk(callback_table_mutex());
        auto it = callback_table().find(id);
        if (it != callback_table().end())
        {
            fn = std::move(it->second);
            callback_table().erase(it);
        }
        return fn;
    }

    inline bool can_use_legion_async()
    {
        return task_registered_flag().load(std::memory_order_acquire) &&
               tls_state().bound &&
               tls_state().runtime != nullptr;
    }

    inline void astm_async_task(
        const Legion::Task* task,
        const std::vector<Legion::PhysicalRegion>&,
        Legion::Context,
        Legion::Runtime*)
    {
        if (task == nullptr || task->args == nullptr || task->arglen != sizeof(async_task_payload))
            return;

        async_task_payload payload;
        std::memcpy(&payload, task->args, sizeof(async_task_payload));

        auto fn = pop_callback(payload.callback_id);
        if (fn)
        {
            transaction* t = reinterpret_cast<transaction*>(payload.transaction_ptr);
            fn(t);
        }
    }

} // namespace detail

// Call once before Runtime::start(...)
inline void preregister_legion_tasks()
{
    static std::once_flag once;
    std::call_once(once, []() {
        Legion::TaskVariantRegistrar registrar(detail::ASTM_INTERNAL_ASYNC_TASK_ID, "astm_internal_async_task");
        Legion::Runtime::preregister_task_variant<detail::astm_async_task>(registrar, "astm_internal_async_task");
        detail::task_registered_flag().store(true, std::memory_order_release);
    });
}

// Call in each Legion task that uses ASTM (typically at top of task body).
inline void bind_legion_context(Legion::Runtime* runtime, Legion::Context context)
{
    detail::tls_state().runtime = runtime;
    detail::tls_state().context = context;
    detail::tls_state().bound = true;
}

inline void unbind_legion_context()
{
    detail::tls_state().runtime = nullptr;
    detail::tls_state().context = Legion::Context();
    detail::tls_state().bound = false;
}

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
    typedef async_future future_type;

  private:
    transaction* trans_;
    future_type fut_;

  public:
    explicit transaction_future(transaction* trans) : trans_(trans), fut_() {}
    explicit transaction_future(transaction& trans) : trans_(&trans), fut_() {}

    template <typename F>
    void then(F f);

    void get() { fut_.get(); }
};

template <typename T>
struct shared_var : shared_var_base
{
    typedef async_future future_type;

    struct local_var
    {
      private:
        transaction* trans_;
        shared_var_base* var_;

      public:
        local_var(transaction* trans, shared_var_base* var) : trans_(trans), var_(var) {}

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

    T const& read() const { return data_; }
    void write(T const& rhs) { data_ = rhs; }

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
    std::list<std::pair<async_future*, std::function<void(transaction*)>>> async_list;
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

        for (auto* var : write_set)
        {
            assert(var != nullptr);
            auto it = variables.find(var);
            assert(it != variables.end());
            var->write(*it->second);
        }

        for (auto& op : async_list)
        {
            async_future* chain = op.first;
            std::function<void(transaction*)> fn = op.second;

            if (detail::can_use_legion_async())
            {
                if (chain && chain->is_std())
                    chain->get();

                detail::async_task_payload payload{
                    detail::store_callback(fn),
                    reinterpret_cast<std::uintptr_t>(this)
                };

                Legion::TaskLauncher launcher(
                    detail::ASTM_INTERNAL_ASYNC_TASK_ID,
                    Legion::TaskArgument(&payload, sizeof(payload)));

                if (chain && chain->is_legion())
                    launcher.add_future(chain->legion_fut);

                Legion::Future f =
                    detail::tls_state().runtime->execute_task(detail::tls_state().context, launcher);

                if (chain)
                    chain->set_legion(std::move(f));
            }
            else
            {
                if (chain == nullptr)
                {
                    std::thread([fn, this]() { fn(this); }).detach();
                }
                else
                {
                    std::shared_future<void> prev;
                    if (chain->is_std())
                        prev = chain->std_fut;
                    else if (chain->is_legion())
                        chain->get();

                    auto sf = std::async(std::launch::async, [prev, fn, this]() mutable {
                        if (prev.valid())
                            prev.wait();
                        fn(this);
                    }).share();

                    chain->set_std(std::move(sf));
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
            read_list.emplace_back(
                var,
                std::shared_ptr<shared_var_base>(result.first->second->clone()));
        }

        return *(result.first->second);
    }

    void write(shared_var_base* var, shared_var_base const& value)
    {
        assert(var != nullptr);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base>> entry(
            var, std::shared_ptr<shared_var_base>(value.clone()));

        auto result = variables.insert(entry);

        if (!result.second)
            result.first->second->write(value);

        write_set.insert(var);
    }

    void then(async_future* fut, std::function<void(transaction*)> F)
    {
        async_list.emplace_back(fut, std::move(F));
    }
};

template <typename T>
shared_var<T>::local_var::operator T const&() const
{
    auto const& base = trans_->read(var_);
    auto const* p = dynamic_cast<shared_var const*>(&base);
    assert(p != nullptr);
    return p->read();
}

template <typename T>
T shared_var<T>::local_var::get() const
{
    auto const& base = trans_->read(var_);
    auto const* p = dynamic_cast<shared_var const*>(&base);
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
    auto* concrete = dynamic_cast<shared_var*>(var_);
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
