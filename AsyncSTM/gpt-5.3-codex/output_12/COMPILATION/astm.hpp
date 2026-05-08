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

#if __has_include(<legion.h>)
  #include <legion.h>
#endif

#include <cassert>
#include <future>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <thread>
#include <type_traits>
#include <utility>

namespace astm
{

namespace detail
{
    // Small future type with continuation support (works well inside Legion tasks
    // without relying on HPX-specific future::then).
    class continuation_future
    {
    public:
        continuation_future()
          : fut_(make_ready_future())
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

            std::packaged_task<void()> task(
                [prev, fn = std::forward<F>(f)]() mutable {
                    prev.wait();
                    fn();
                });

            auto next = task.get_future().share();
            std::thread(std::move(task)).detach();
            return continuation_future(std::move(next));
        }

    private:
        static std::shared_future<void> make_ready_future()
        {
            std::promise<void> p;
            p.set_value();
            return p.get_future().share();
        }

        std::shared_future<void> fut_;
    };
} // namespace detail

struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;
    virtual void write(shared_var_base const&) = 0;
    virtual ASTM_LOCK lock() const = 0;
    virtual bool operator==(shared_var_base const&) const = 0;
};

struct transaction;

struct transaction_future
{
    typedef detail::continuation_future future_type;

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
    typedef detail::continuation_future future_type;

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

    // NOTE: direct read/write are intentionally unlocked like original code.
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
    using future_type = detail::continuation_future;

    std::list<
        std::pair<
            shared_var_base*,                   // shared variable read from
            std::shared_ptr<shared_var_base>    // snapshot value read
        >
    > read_list;

    std::set<shared_var_base*> write_set;

    std::list<
        std::pair<
            future_type*,                       // null => fire-and-forget
            ASTM_FUNCTION<void(transaction*)>   // async action
        >
    > async_list;

    std::map<
        shared_var_base*,
        std::shared_ptr<shared_var_base>        // transaction-local current value
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
        // 1) Lock all touched variables in sorted order (map order).
        std::list<ASTM_LOCK> locks;
        for (auto const& var : variables)
        {
            assert(var.first != nullptr);
            locks.push_back(var.first->lock());
        }

        // 2) Validate read-set snapshots.
        for (auto const& var : read_list)
        {
            assert(var.first != nullptr);
            if (!((*var.first) == (*var.second)))
            {
                clear();
                return false;
            }
        }

        // 3) Apply writes from transaction-local state.
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
            {
                auto fn = op.second;
                std::thread([fn, this]() mutable { fn(this); }).detach();
            }
            else
            {
                auto fn = op.second;
                *op.first = op.first->then([fn, this]() mutable { fn(this); });
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
            // First touch: clone current shared value into local state.
            result.first->second.reset(var->clone());

            // Keep an independent snapshot for validation.
            read_list.push_back(
                std::make_pair(
                    var,
                    std::shared_ptr<shared_var_base>(result.first->second->clone())));
        }

        return *(result.first->second);
    }

    void write(shared_var_base* var, shared_var_base const& value)
    {
        assert(var != nullptr);

        auto result = variables.emplace(
            var, std::shared_ptr<shared_var_base>(value.clone()));

        if (!result.second)
        {
            // Update existing local value without replacing container node.
            result.first->second->write(value);
        }

        write_set.insert(var);
    }

    void then(future_type* fut, ASTM_FUNCTION<void(transaction*)> F)
    {
        async_list.push_back(std::make_pair(fut, std::move(F)));
    }
};

template <typename T>
shared_var<T>::local_var::operator T const& () const
{
    assert(trans_ != nullptr);
    assert(var_ != nullptr);

    auto const& base = trans_->read(var_);
    auto const* typed = dynamic_cast<shared_var const*>(&base);
    assert(typed != nullptr);
    return typed->read();
}

template <typename T>
T shared_var<T>::local_var::get() const
{
    assert(trans_ != nullptr);
    assert(var_ != nullptr);

    auto const& base = trans_->read(var_);
    auto const* typed = dynamic_cast<shared_var const*>(&base);
    assert(typed != nullptr);
    return typed->read();
}

template <typename T>
typename shared_var<T>::local_var&
shared_var<T>::local_var::operator=(shared_var_base const& rhs)
{
    assert(trans_ != nullptr);
    assert(var_ != nullptr);
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
    assert(var_ != nullptr);

    auto* typed = dynamic_cast<shared_var*>(var_);
    assert(typed != nullptr);

    trans_->then(&typed->queue, ASTM_FUNCTION<void(transaction*)>(std::move(f)));
}

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_ != nullptr);
    trans_->then(&fut_, ASTM_FUNCTION<void(transaction*)>(std::move(f)));
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
