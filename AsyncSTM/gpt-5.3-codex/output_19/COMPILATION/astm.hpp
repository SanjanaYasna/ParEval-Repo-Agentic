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
#include <cstring>
#include <future>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace astm
{

struct transaction;

namespace detail
{
    // One internal Legion task used to run continuation closures.
    constexpr Legion::TaskID ASTM_LEGION_CONTINUATION_TASK_ID = 1000001;

    class continuation_registry
    {
    public:
        uint64_t emplace(std::function<void()> fn)
        {
            std::lock_guard<std::mutex> lk(mtx_);
            const uint64_t id = next_id_++;
            table_.emplace(id, std::move(fn));
            return id;
        }

        std::function<void()> take(uint64_t id)
        {
            std::lock_guard<std::mutex> lk(mtx_);
            auto it = table_.find(id);
            if (it == table_.end())
                return {};
            auto fn = std::move(it->second);
            table_.erase(it);
            return fn;
        }

        static continuation_registry& instance()
        {
            static continuation_registry r;
            return r;
        }

    private:
        std::mutex mtx_;
        std::unordered_map<uint64_t, std::function<void()>> table_;
        uint64_t next_id_ = 1;
    };

    inline void astm_legion_continuation_task(
        Legion::Task const* task,
        std::vector<Legion::PhysicalRegion> const&,
        Legion::Context,
        Legion::Runtime*)
    {
        assert(task != nullptr);
        assert(task->arglen == sizeof(uint64_t));

        uint64_t token = 0;
        std::memcpy(&token, task->args, sizeof(uint64_t));

        auto fn = continuation_registry::instance().take(token);
        if (fn) fn();
    }

    inline void preregister_continuation_task()
    {
        static bool registered = []() -> bool {
            Legion::TaskVariantRegistrar registrar(
                ASTM_LEGION_CONTINUATION_TASK_ID,
                "astm_legion_continuation_task");
            registrar.add_constraint(
                Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
            registrar.set_leaf();
            Legion::Runtime::preregister_task_variant<astm_legion_continuation_task>(
                registrar, "astm_legion_continuation_task");
            return true;
        }();
        (void)registered;
    }

    // Force preregistration before Runtime::start.
    inline const bool continuation_task_preregistered = []() -> bool {
        preregister_continuation_task();
        return true;
    }();

    inline thread_local Legion::Runtime* tls_runtime = nullptr;
    inline thread_local Legion::Context tls_context{};
    inline thread_local bool tls_has_context = false;

    inline void bind_context(Legion::Runtime* runtime, Legion::Context context)
    {
        (void)continuation_task_preregistered;
        tls_runtime = runtime;
        tls_context = context;
        tls_has_context = (runtime != nullptr);
    }

    inline void unbind_context()
    {
        tls_runtime = nullptr;
        tls_has_context = false;
    }

    inline bool has_bound_context()
    {
        return tls_has_context && (tls_runtime != nullptr);
    }

    struct scheduled_async
    {
        enum class backend { none, legion, std_thread };

        backend kind = backend::none;
        Legion::Future legion_future;
        std::shared_future<void> std_future;

        static scheduled_async from_legion(Legion::Future f)
        {
            scheduled_async out;
            out.kind = backend::legion;
            out.legion_future = std::move(f);
            return out;
        }

        static scheduled_async from_std(std::shared_future<void> f)
        {
            scheduled_async out;
            out.kind = backend::std_thread;
            out.std_future = std::move(f);
            return out;
        }

        void wait()
        {
            switch (kind)
            {
            case backend::legion:
                legion_future.get_void_result();
                break;
            case backend::std_thread:
                std_future.wait();
                break;
            case backend::none:
            default:
                break;
            }
        }
    };

    inline scheduled_async launch_tracked(std::function<void()> fn)
    {
        if (has_bound_context())
        {
            (void)continuation_task_preregistered;

            const uint64_t token = continuation_registry::instance().emplace(std::move(fn));
            Legion::TaskArgument arg(&token, sizeof(token));
            Legion::TaskLauncher launcher(ASTM_LEGION_CONTINUATION_TASK_ID, arg);
            launcher.enable_inlining = true; // keep closure execution local/safe
            Legion::Future f = tls_runtime->execute_task(tls_context, launcher);
            return scheduled_async::from_legion(std::move(f));
        }

        // Fallback when called outside a Legion task context.
        auto task = std::make_shared<std::packaged_task<void()>>(std::move(fn));
        std::shared_future<void> sf = task->get_future().share();
        std::thread([task]() { (*task)(); }).detach();
        return scheduled_async::from_std(std::move(sf));
    }

    inline void launch_fire_and_forget(std::function<void()> fn)
    {
        if (has_bound_context())
        {
            (void)continuation_task_preregistered;

            const uint64_t token = continuation_registry::instance().emplace(std::move(fn));
            Legion::TaskArgument arg(&token, sizeof(token));
            Legion::TaskLauncher launcher(ASTM_LEGION_CONTINUATION_TASK_ID, arg);
            launcher.enable_inlining = true;
            (void)tls_runtime->execute_task(tls_context, launcher);
            return;
        }

        std::thread([f = std::move(fn)]() mutable { f(); }).detach();
    }

    struct continuation_handle
    {
        void append(scheduled_async s)
        {
            std::lock_guard<std::mutex> lk(mtx_);
            futures_.push_back(std::move(s));
        }

        void wait_all()
        {
            std::vector<scheduled_async> local;
            {
                std::lock_guard<std::mutex> lk(mtx_);
                local.swap(futures_);
            }
            for (auto& f : local) f.wait();
        }

    private:
        std::mutex mtx_;
        std::vector<scheduled_async> futures_;
    };
} // namespace detail

// Call this at the beginning of a Legion task to enable Legion-backed continuations.
inline void bind_legion_context(Legion::Runtime* runtime, Legion::Context context)
{
    detail::bind_context(runtime, context);
}

inline void unbind_legion_context()
{
    detail::unbind_context();
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
  private:
    transaction* trans_;
    std::shared_ptr<detail::continuation_handle> handle_;

  public:
    explicit transaction_future(transaction* trans)
      : trans_(trans), handle_(std::make_shared<detail::continuation_handle>())
    {}

    explicit transaction_future(transaction& trans)
      : trans_(&trans), handle_(std::make_shared<detail::continuation_handle>())
    {}

    template <typename F>
    void then(F f);

    void get()
    {
        handle_->wait_all();
    }

    detail::continuation_handle* native_handle()
    {
        return handle_.get();
    }
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
    detail::continuation_handle queue;

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
    std::list<std::pair<detail::continuation_handle*, std::function<void(transaction*)>>> async_list;
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

        // 1) Lock all touched variables in sorted order (map order).
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

        // 4) Schedule async continuations.
        for (auto& op : async_list)
        {
            auto thunk = [fn = op.second, this]() mutable { fn(this); };
            if (op.first == nullptr)
            {
                detail::launch_fire_and_forget(std::move(thunk));
            }
            else
            {
                op.first->append(detail::launch_tracked(std::move(thunk)));
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

    void then(detail::continuation_handle* fut, std::function<void(transaction*)> F)
    {
        async_list.emplace_back(fut, std::move(F));
    }
};

template <typename T>
shared_var<T>::local_var::operator T const& () const
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
    auto* owner = dynamic_cast<shared_var*>(var_);
    assert(owner != nullptr);
    trans_->then(&owner->queue, std::function<void(transaction*)>(std::move(f)));
}

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_ != nullptr);
    trans_->then(native_handle(), std::function<void(transaction*)>(std::move(f)));
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
