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
#include <list>
#include <map>
#include <memory>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

namespace astm
{

struct transaction;

struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;
    virtual void write(shared_var_base const&) = 0;
    virtual ASTM_LOCK lock() const = 0;
    virtual bool operator==(shared_var_base const&) const = 0;
};

namespace detail
{
    using continuation_fn = ASTM_FUNCTION<void(transaction*)>;

    template <typename>
    inline constexpr bool always_false_v = false;

    template <typename F>
    continuation_fn make_continuation(F&& f)
    {
        using Fn = std::decay_t<F>;
        return [fn = Fn(std::forward<F>(f))](transaction* t) mutable {
            if constexpr (std::is_invocable_v<Fn&, transaction*>)
            {
                fn(t);
            }
            else if constexpr (std::is_invocable_v<Fn&>)
            {
                fn();
            }
            else
            {
                static_assert(always_false_v<Fn>,
                    "Continuation must be callable with (transaction*) or with no arguments");
            }
        };
    }

    struct async_handle
    {
        std::shared_future<void> fut;

        async_handle() = default;

        explicit async_handle(std::future<void>&& f)
          : fut(f.share())
        {}

        void wait()
        {
            if (fut.valid()) fut.wait();
        }
    };

    struct continuation_future
    {
        std::vector<async_handle> handles;

        void add(async_handle h)
        {
            handles.push_back(std::move(h));
        }

        void get()
        {
            for (auto& h : handles)
                h.wait();
            handles.clear();
        }
    };

    inline async_handle spawn_tracked(continuation_fn fn, transaction* trans)
    {
        return async_handle(std::async(
            std::launch::async,
            [op = std::move(fn), trans]() mutable { op(trans); }));
    }

    inline void spawn_detached(continuation_fn fn, transaction* trans)
    {
        std::thread([op = std::move(fn), trans]() mutable { op(trans); }).detach();
    }

} // namespace detail

template <typename RuntimeT, typename ContextT>
inline void bind_legion_runtime(RuntimeT*, ContextT)
{
}

inline void unbind_legion_runtime()
{
}

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
    void then(F f);

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

        operator T() const;

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
      : data_(rhs.read())
      , mtx_()
      , queue()
    {}

    ~shared_var() {}

    shared_var_base* clone() const override
    {
        ASTM_LOCK l(mtx_);
        return new shared_var(data_);
    }

    T read() const
    {
        ASTM_LOCK l(mtx_);
        return data_;
    }

    void write(T const& rhs)
    {
        ASTM_LOCK l(mtx_);
        data_ = rhs;
    }

    void write(shared_var_base const& rhs) override
    {
        auto const* typed = dynamic_cast<shared_var const*>(&rhs);
        assert(typed != nullptr);
        data_ = typed->read();
    }

    ASTM_LOCK lock() const override
    {
        return ASTM_LOCK(mtx_);
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
    using future_type = detail::continuation_future;
    using continuation_fn = detail::continuation_fn;

    std::list<
        std::pair<
            shared_var_base*,
            std::shared_ptr<shared_var_base>>>
        read_list;

    std::set<shared_var_base*> write_set;

    std::list<
        std::pair<
            future_type*,
            continuation_fn>>
        async_list;

    std::map<
        shared_var_base*,
        std::shared_ptr<shared_var_base>>
        variables;

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

        for (shared_var_base* var : write_set)
        {
            assert(var != nullptr);
            auto it = variables.find(var);
            assert(it != variables.end());
            (*var).write(*((*it).second));
        }

        for (auto& op : async_list)
        {
            if (op.first == nullptr)
            {
                detail::spawn_detached(op.second, this);
            }
            else
            {
                op.first->add(detail::spawn_tracked(op.second, this));
            }
        }

        return true;
    }

    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != nullptr);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base>> entry(var, nullptr);
        auto result = variables.insert(entry);

        if (result.second == true)
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

        if (result.second == true)
        {
            write_set.insert(var);
        }
        else
        {
            result.first->second = entry.second;
            write_set.insert(var);
        }
    }

    void then(future_type* fut, continuation_fn F)
    {
        async_list.emplace_back(fut, std::move(F));
    }
};

template <typename T>
shared_var<T>::local_var::operator T() const
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
    trans_->then(&concrete->queue, detail::make_continuation(std::move(f)));
}

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_ != nullptr);
    trans_->then(&fut_, detail::make_continuation(std::move(f)));
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
