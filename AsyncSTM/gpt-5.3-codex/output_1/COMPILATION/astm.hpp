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
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace astm
{

namespace detail
{
    inline ASTM_FUTURE<void> launch_async(std::function<void()> fn)
    {
        return ASTM_ASYNC([f = std::move(fn)]() mutable { f(); });
    }
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
    using future_type = ASTM_FUTURE<void>;

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
        if (fut_.valid())
            fut_.get();
    }
};

template <typename T>
struct shared_var : shared_var_base
{
    using future_type = ASTM_FUTURE<void>;

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

    ~shared_var() override = default;

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
        data_ = dynamic_cast<shared_var<T> const*>(&rhs)->read();
    }

    ASTM_LOCK lock() const override
    {
        return ASTM_LOCK(mtx_);
    }

    bool operator==(shared_var_base const& rhs) const override
    {
        return data_ == dynamic_cast<shared_var<T> const*>(&rhs)->read();
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
            ASTM_FUTURE<void>*,
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
        // 1) Lock all variables in sorted order.
        std::list<ASTM_LOCK> locks;
        for (auto const& var : variables)
        {
            assert(var.first != nullptr);
            locks.emplace_back((*var.first).lock());
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
            (*var).write((*(*it).second));
        }

        // 4) Execute async continuations.
        for (auto& op : async_list)
        {
            if (op.first == nullptr)
            {
                (void)detail::launch_async([fn = op.second, this]() mutable {
                    fn(this);
                });
            }
            else
            {
                if (op.first->valid())
                {
                    auto prev = *op.first;
                    *op.first = detail::launch_async([fn = op.second, prev, this]() mutable {
                        prev.get();
                        fn(this);
                    });
                }
                else
                {
                    *op.first = detail::launch_async([fn = op.second, this]() mutable {
                        fn(this);
                    });
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
            (*result.first).second.reset((*var).clone());
            read_list.push_back(*result.first);
            return (*(*result.first).second);
        }

        return (*(*result.first).second);
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
            (*result.first).second = entry.second;
            write_set.insert(var);
        }
    }

    void then(ASTM_FUTURE<void>* fut, std::function<void(transaction*)> F)
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
    auto* typed = dynamic_cast<shared_var<T>*>(var_);
    assert(typed != nullptr);

    trans_->then(
        &typed->queue,
        [fn = std::function<void(transaction*)>(std::move(f))](transaction* tr) mutable {
            fn(tr);
        });
}

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_ != nullptr);

    trans_->then(
        &fut_,
        [fn = std::function<void(transaction*)>(std::move(f))](transaction* tr) mutable {
            fn(tr);
        });
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
