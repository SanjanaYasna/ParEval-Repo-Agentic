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

#include <cassert>
#include <cstdint>
#include <exception>
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

struct transaction;

namespace detail
{
    using continuation_fn = std::function<void(transaction*)>;

    inline std::shared_future<void> make_ready_shared_future()
    {
        std::promise<void> p;
        p.set_value();
        return p.get_future().share();
    }

    struct continuation_chain
    {
        std::mutex mtx;
        std::shared_future<void> tail;

        continuation_chain()
          : tail(make_ready_shared_future())
        {}
    };

    inline void schedule_continuation(
        transaction* trans,
        continuation_chain* chain,
        continuation_fn fn)
    {
        if (chain == nullptr)
        {
            std::thread([fn = std::move(fn), trans]() mutable {
                try
                {
                    fn(trans);
                }
                catch (...)
                {
                }
            }).detach();
            return;
        }

        auto completion = std::make_shared<std::promise<void>>();
        std::shared_future<void> prev;

        {
            std::lock_guard<std::mutex> guard(chain->mtx);
            prev = chain->tail;
            chain->tail = completion->get_future().share();
        }

        std::thread([prev, completion, fn = std::move(fn), trans]() mutable {
            try
            {
                prev.get();
                fn(trans);
                completion->set_value();
            }
            catch (...)
            {
                try
                {
                    completion->set_exception(std::current_exception());
                }
                catch (...)
                {
                }
            }
        }).detach();
    }

} // namespace detail

template <typename RuntimeT, typename ContextT>
inline void bind_legion_context(RuntimeT*, ContextT)
{
}

inline void unbind_legion_context()
{
}

struct shared_var_base
{
    using lock_type = std::unique_lock<std::mutex>;

    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;
    virtual void write(shared_var_base const&) = 0;
    virtual lock_type lock() const = 0;
    virtual bool operator==(shared_var_base const&) const = 0;
};

struct transaction_future
{
  private:
    transaction* trans_;
    std::shared_ptr<detail::continuation_chain> chain_;

  public:
    explicit transaction_future(transaction* trans)
      : trans_(trans)
      , chain_(std::make_shared<detail::continuation_chain>())
    {}

    explicit transaction_future(transaction& trans)
      : trans_(&trans)
      , chain_(std::make_shared<detail::continuation_chain>())
    {}

    template <typename F>
    void then(F f);

    void get()
    {
        std::shared_future<void> wait_for;
        {
            std::lock_guard<std::mutex> guard(chain_->mtx);
            wait_for = chain_->tail;
        }
        wait_for.get();
    }

    detail::continuation_chain* chain() const
    {
        return chain_.get();
    }
};

template <typename T>
struct shared_var : shared_var_base
{
    using continuation_chain_ptr = std::shared_ptr<detail::continuation_chain>;

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
    mutable std::mutex mtx_;

  public:
    continuation_chain_ptr queue;

    shared_var()
      : data_()
      , mtx_()
      , queue(std::make_shared<detail::continuation_chain>())
    {}

    explicit shared_var(T const& t)
      : data_(t)
      , mtx_()
      , queue(std::make_shared<detail::continuation_chain>())
    {}

    explicit shared_var(T&& t)
      : data_(std::move(t))
      , mtx_()
      , queue(std::make_shared<detail::continuation_chain>())
    {}

    shared_var(shared_var const& rhs)
      : data_(rhs.data_)
      , mtx_()
      , queue(std::make_shared<detail::continuation_chain>())
    {}

    ~shared_var() {}

    shared_var_base* clone() const override
    {
        auto l = lock();
        (void) l;
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

    lock_type lock() const override
    {
        return lock_type(mtx_);
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
            std::shared_ptr<shared_var_base>
        >
    > read_list;

    std::set<shared_var_base*> write_set;

    std::list<
        std::pair<
            detail::continuation_chain*,
            detail::continuation_fn
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
        std::list<shared_var_base::lock_type> locks;

        for (auto const& var : variables)
        {
            assert(var.first != nullptr);
            locks.push_back(var.first->lock());
        }

        for (auto const& r : read_list)
        {
            assert(r.first != nullptr);
            if (!(*(r.first) == *(r.second)))
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
            detail::schedule_continuation(this, op.first, op.second);
        }

        return true;
    }

    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != nullptr);

        auto res = variables.insert(std::make_pair(var, std::shared_ptr<shared_var_base>()));
        if (res.second)
        {
            res.first->second.reset(var->clone());

            read_list.push_back(std::make_pair(
                var,
                std::shared_ptr<shared_var_base>(res.first->second->clone())));
        }

        return *(res.first->second);
    }

    void write(shared_var_base* var, shared_var_base const& value)
    {
        assert(var != nullptr);

        auto res = variables.insert(std::make_pair(var, std::shared_ptr<shared_var_base>()));
        if (res.second)
        {
            res.first->second.reset(value.clone());
        }
        else
        {
            res.first->second->write(value);
        }

        write_set.insert(var);
    }

    void then(detail::continuation_chain* chain, detail::continuation_fn f)
    {
        async_list.push_back(std::make_pair(chain, std::move(f)));
    }
};

template <typename T>
shared_var<T>::local_var::operator T const& () const
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

    auto* owner = dynamic_cast<shared_var<T>*>(var_);
    assert(owner != nullptr);

    trans_->then(owner->queue.get(), detail::continuation_fn(std::move(f)));
}

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_ != nullptr);
    trans_->then(chain_.get(), detail::continuation_fn(std::move(f)));
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
