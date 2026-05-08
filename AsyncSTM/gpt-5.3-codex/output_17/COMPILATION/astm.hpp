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
#include <future>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

namespace astm
{

using mutex_type = std::mutex;
using lock_type  = std::unique_lock<mutex_type>;

struct transaction;

namespace detail
{
    inline std::shared_future<void> make_ready_shared_future()
    {
        std::promise<void> p;
        p.set_value();
        return p.get_future().share();
    }

    class continuation_future
    {
    public:
        continuation_future()
          : fut_(make_ready_shared_future())
        {}

        explicit continuation_future(std::shared_future<void> f)
          : fut_(std::move(f))
        {}

        void get() const
        {
            fut_.wait();
        }

        template <typename F>
        continuation_future then(F&& f) const
        {
            auto prev = fut_;
            auto next = std::async(
                std::launch::async,
                [prev, fn = std::forward<F>(f)]() mutable {
                    prev.wait();
                    fn();
                }).share();

            return continuation_future(std::move(next));
        }

    private:
        std::shared_future<void> fut_;
    };

    template <typename>
    struct always_false : std::false_type
    {};

    template <typename F>
    std::function<void(transaction*)> make_action(F&& f)
    {
        using Fn = std::decay_t<F>;

        if constexpr (std::is_invocable_v<Fn, transaction*>)
        {
            return [fn = Fn(std::forward<F>(f))](transaction* t) mutable {
                fn(t);
            };
        }
        else if constexpr (std::is_invocable_v<Fn>)
        {
            return [fn = Fn(std::forward<F>(f))](transaction*) mutable {
                fn();
            };
        }
        else
        {
            static_assert(always_false<Fn>::value,
                "Continuation must be invocable as f() or f(transaction*)");
        }
    }

} // namespace detail

struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;
    virtual void write(shared_var_base const&) = 0;
    virtual lock_type lock() const = 0;
    virtual bool operator==(shared_var_base const&) const = 0;
};

struct transaction_future
{
    using future_type = detail::continuation_future;

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
    void then(F&& f);

    void get()
    {
        fut_.get();
    }
};

template <typename T>
struct shared_var : shared_var_base
{
    using future_type = detail::continuation_future;

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
    mutable mutex_type mtx_;

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

    ~shared_var() override = default;

    shared_var_base* clone() const override
    {
        auto l = lock();
        (void)l;
        return new shared_var(data_);
    }

    // Doesn't lock (same behavior as original code).
    T const& read() const
    {
        return data_;
    }

    // Doesn't lock (same behavior as original code).
    void write(T const& rhs)
    {
        data_ = rhs;
    }

    // Doesn't lock (same behavior as original code).
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
    using future_type = detail::continuation_future;
    using async_function_type = std::function<void(transaction*)>;

    std::list<std::pair<shared_var_base*, std::shared_ptr<shared_var_base>>> read_list;
    std::set<shared_var_base*> write_set;
    std::list<std::pair<future_type*, async_function_type>> async_list;
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
        // 1) Lock all variables in sorted order (map key order).
        std::list<lock_type> locks;
        for (auto const& var : variables)
        {
            assert(var.first != nullptr);
            locks.push_back(var.first->lock());
        }

        // 2) Validate read-set.
        for (auto const& var : read_list)
        {
            assert(var.first != nullptr);
            if (!(*(var.first) == *(var.second)))
            {
                clear();
                return false;
            }
        }

        // 3) Apply writes.
        for (auto* var : write_set)
        {
            assert(var != nullptr);
            auto it = variables.find(var);
            assert(it != variables.end());
            var->write(*(it->second));
        }

        // 4) Execute continuations.
        for (auto& op : async_list)
        {
            if (op.first == nullptr)
            {
                auto ignored = std::async(
                    std::launch::async,
                    [fn = op.second, this]() mutable { fn(this); });
                (void)ignored;
            }
            else
            {
                *(op.first) = op.first->then(
                    [fn = op.second, this]() mutable { fn(this); });
            }
        }

        // 5) Locks released by RAII.
        return true;
    }

    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != nullptr);

        auto result = variables.emplace(var, std::shared_ptr<shared_var_base>());

        if (result.second)
        {
            auto snapshot = std::shared_ptr<shared_var_base>(var->clone());
            result.first->second = snapshot;
            read_list.emplace_back(var, snapshot);
            return *(result.first->second);
        }

        return *(result.first->second);
    }

    void write(shared_var_base* var, shared_var_base const& value)
    {
        assert(var != nullptr);

        auto cloned = std::shared_ptr<shared_var_base>(value.clone());
        auto result = variables.emplace(var, cloned);
        if (!result.second)
            result.first->second = std::move(cloned);

        write_set.insert(var);
    }

    // If fut is nullptr, fire-and-forget semantics are used.
    void then(future_type* fut, async_function_type F)
    {
        async_list.emplace_back(fut, std::move(F));
    }
};

template <typename T>
shared_var<T>::local_var::operator T const&() const
{
    auto const& ref = trans_->read(var_);
    auto const* p = dynamic_cast<shared_var const*>(&ref);
    assert(p != nullptr);
    return p->read();
}

template <typename T>
T shared_var<T>::local_var::get() const
{
    auto const& ref = trans_->read(var_);
    auto const* p = dynamic_cast<shared_var const*>(&ref);
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
void shared_var<T>::local_var::then(F&& f)
{
    assert(trans_ != nullptr);
    auto* concrete = dynamic_cast<shared_var<T>*>(var_);
    assert(concrete != nullptr);
    trans_->then(&concrete->queue, detail::make_action(std::forward<F>(f)));
}

template <typename F>
void transaction_future::then(F&& f)
{
    assert(trans_ != nullptr);
    trans_->then(&fut_, detail::make_action(std::forward<F>(f)));
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
