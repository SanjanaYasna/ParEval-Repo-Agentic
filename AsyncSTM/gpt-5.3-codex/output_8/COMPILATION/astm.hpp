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
#include <functional>
#include <future>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
#include <vector>

namespace astm
{

// NOTE:
// This Legion-oriented ASTM variant keeps STM metadata host-local and thread-safe,
// so it can be used inside Legion tasks (default mapper, no custom mapping needed).

inline std::shared_future<void> make_ready_future()
{
    std::promise<void> p;
    p.set_value();
    return p.get_future().share();
}

template <typename F>
inline std::shared_future<void> chain_future(std::shared_future<void> predecessor, F&& f)
{
    auto task = std::make_shared<std::packaged_task<void()>>(
        [predecessor, fn = std::function<void()>(std::forward<F>(f))]() mutable {
            predecessor.wait();
            fn();
        });

    std::shared_future<void> out = task->get_future().share();
    std::thread([task]() { (*task)(); }).detach();
    return out;
}

template <typename F>
inline void launch_detached(F&& f)
{
    std::thread(std::forward<F>(f)).detach();
}

struct shared_var_base
{
    using lock_type = std::unique_lock<std::mutex>;

    virtual ~shared_var_base() = default;

    virtual shared_var_base* clone() const = 0;
    virtual void write(shared_var_base const&) = 0;
    virtual lock_type lock() const = 0;
    virtual bool operator==(shared_var_base const&) const = 0;
};

struct transaction;

struct transaction_future
{
    using future_type = std::shared_future<void>;

  private:
    transaction* trans_;
    future_type fut_;

  public:
    explicit transaction_future(transaction* trans)
      : trans_(trans)
      , fut_(make_ready_future())
    {}

    explicit transaction_future(transaction& trans)
      : trans_(&trans)
      , fut_(make_ready_future())
    {}

    template <typename F>
    void then(F&& f);

    void get()
    {
        fut_.wait();
    }
};

template <typename T>
struct shared_var : shared_var_base
{
    using future_type = std::shared_future<void>;

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
    mutable std::mutex mtx_;

  public:
    future_type queue;

    shared_var()
      : data_()
      , mtx_()
      , queue(make_ready_future())
    {}

    explicit shared_var(T const& t)
      : data_(t)
      , mtx_()
      , queue(make_ready_future())
    {}

    explicit shared_var(T&& t)
      : data_(std::move(t))
      , mtx_()
      , queue(make_ready_future())
    {}

    shared_var(shared_var const& rhs)
      : data_()
      , mtx_()
      , queue(make_ready_future())
    {
        auto l = rhs.lock();
        data_ = rhs.data_;
    }

    ~shared_var() override = default;

    shared_var_base* clone() const override
    {
        auto l = lock();
        return new shared_var(data_);
    }

    // Doesn't lock (intentionally; caller controls synchronization).
    T const& read() const
    {
        return data_;
    }

    // Doesn't lock (intentionally; caller controls synchronization).
    void write(T const& rhs)
    {
        data_ = rhs;
    }

    // Doesn't lock (intentionally; caller controls synchronization).
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
    using future_type = std::shared_future<void>;
    using async_function_type = std::function<void(transaction*)>;

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

    // Optional Legion handles for integration in caller code paths.
    Legion::Runtime* runtime{nullptr};
    Legion::Context context{};

    transaction() = default;

    transaction(Legion::Runtime* rt, Legion::Context ctx)
      : runtime(rt)
      , context(ctx)
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
        // 1) Lock all variables in deterministic order (map key order).
        std::list<shared_var_base::lock_type> locks;
        for (auto const& var : variables)
        {
            assert(var.first != nullptr);
            locks.push_back(var.first->lock());
        }

        // 2) Validate read set.
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
            var->write(*(it->second));
        }

        // 4) Schedule async continuations.
        for (auto& op : async_list)
        {
            if (op.first == nullptr)
            {
                launch_detached([fn = op.second, this]() mutable { fn(this); });
            }
            else
            {
                *(op.first) = chain_future(
                    *(op.first),
                    [fn = op.second, this]() mutable { fn(this); });
            }
        }

        // 5) Locks released by RAII.
        return true;
    }

    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != nullptr);

        auto result = variables.insert({var, std::shared_ptr<shared_var_base>()});
        auto it = result.first;
        bool inserted = result.second;

        if (inserted)
        {
            it->second.reset(var->clone());
            read_list.push_back(*it);
        }

        return *(it->second);
    }

    void write(shared_var_base* var, shared_var_base const& value)
    {
        assert(var != nullptr);

        auto cloned = std::shared_ptr<shared_var_base>(value.clone());
        auto result = variables.insert({var, cloned});
        auto it = result.first;
        bool inserted = result.second;

        if (!inserted)
            it->second = std::move(cloned);

        write_set.insert(var);
    }

    template <typename F>
    void then(future_type* fut, F&& fn)
    {
        async_list.emplace_back(
            fut, async_function_type(std::forward<F>(fn)));
    }
};

template <typename T>
shared_var<T>::local_var::operator T const&() const
{
    auto const& v = trans_->read(var_);
    auto const* p = dynamic_cast<shared_var const*>(&v);
    assert(p != nullptr);
    return p->read();
}

template <typename T>
T shared_var<T>::local_var::get() const
{
    auto const& v = trans_->read(var_);
    auto const* p = dynamic_cast<shared_var const*>(&v);
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
void shared_var<T>::local_var::then(F&& f)
{
    assert(trans_ != nullptr);
    auto* owner = dynamic_cast<shared_var<T>*>(var_);
    assert(owner != nullptr);
    trans_->then(&owner->queue, std::forward<F>(f));
}

template <typename F>
void transaction_future::then(F&& f)
{
    assert(trans_ != nullptr);
    trans_->then(&fut_, std::forward<F>(f));
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
