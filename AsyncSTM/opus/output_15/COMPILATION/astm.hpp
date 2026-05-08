////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94

#include "legion.h"

#include <list>
#include <set>
#include <map>
#include <vector>
#include <functional>
#include <memory>
#include <cassert>
#include <mutex>
#include <future>
#include <thread>

namespace astm
{

using namespace Legion;

// Field ID used for shared variable data stored in Legion logical regions.
enum ASTMFieldIDs
{
    FID_VAL = 101,
};

// ---------------------------------------------------------------------------
// Thread-local Legion Runtime* and Context, set at the entry of every
// Legion task so that shared_var objects created inside that task can
// allocate logical regions.
// ---------------------------------------------------------------------------
inline Runtime*& current_runtime()
{
    static thread_local Runtime* rt = nullptr;
    return rt;
}

inline Context& current_context()
{
    static thread_local Context ctx;
    return ctx;
}

inline void set_legion_context(Runtime* rt, Context ctx)
{
    current_runtime() = rt;
    current_context()  = ctx;
}

// ---------------------------------------------------------------------------
// A lightweight future that supports .then() chaining, built on top of
// std::shared_future<void>.  Legion::Future does not expose .then(), so
// we keep our own wrapper that is compatible with the rest of the STM
// machinery.
// ---------------------------------------------------------------------------
struct astm_future
{
    std::shared_future<void> fut_;

    astm_future()
    {
        std::promise<void> p;
        p.set_value();
        fut_ = p.get_future().share();
    }

    explicit astm_future(std::shared_future<void> f) : fut_(std::move(f)) {}
    explicit astm_future(std::future<void>&& f)      : fut_(f.share()) {}

    astm_future(astm_future const&)            = default;
    astm_future(astm_future&&)                 = default;
    astm_future& operator=(astm_future const&) = default;
    astm_future& operator=(astm_future&&)      = default;

    template <typename F>
    astm_future then(F f)
    {
        auto prev = fut_;
        return astm_future(
            std::async(std::launch::async,
                       [prev, f]() mutable { prev.wait(); f(); }).share());
    }

    void get() { fut_.get(); }
};

// ---------------------------------------------------------------------------
// shared_var_base  –  type-erased interface for transactional variables
// ---------------------------------------------------------------------------
struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;

    virtual void write(shared_var_base const&) = 0;

    virtual std::unique_lock<std::mutex> lock() const = 0;

    virtual bool operator==(shared_var_base const&) const = 0;
};

struct transaction;

// ---------------------------------------------------------------------------
// transaction_future  –  a future whose continuation is registered with
//                        a transaction and fires after a successful commit
// ---------------------------------------------------------------------------
struct transaction_future
{
    typedef astm_future future_type;

  private:
    transaction* trans_;
    future_type  fut_;

  public:
    transaction_future(transaction* trans)
      : trans_(trans)
      , fut_()
    {}

    transaction_future(transaction& trans)
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

// ---------------------------------------------------------------------------
// shared_var<T>  –  a transactional variable backed by a Legion logical
//                   region (one point, one field of type T).
//
// The in-memory field  data_  is the authoritative copy used by the STM
// protocol; the logical region mirrors it and can be passed to Legion
// child tasks via RegionRequirements.
// ---------------------------------------------------------------------------
template <typename T>
struct shared_var : shared_var_base
{
    typedef astm_future future_type;

    // ----- local_var (transaction-scoped proxy) ----------------------------
    struct local_var
    {
      private:
        transaction*    trans_;
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
    T                data_;
    mutable std::mutex mtx_;

    // Legion region that mirrors data_ (one-point index space, one field).
    LogicalRegion lr_;
    IndexSpace    is_;
    FieldSpace    fs_;
    bool          owns_region_;

    // Tag type to create snapshot copies that skip region allocation.
    struct snapshot_tag {};

    shared_var(T const& t, snapshot_tag)
      : data_(t)
      , mtx_()
      , lr_(LogicalRegion::NO_REGION)
      , is_(IndexSpace::NO_SPACE)
      , fs_(FieldSpace::NO_SPACE)
      , owns_region_(false)
      , queue()
    {}

    // -- helpers to manage the backing Legion region ------------------------

    void create_legion_region()
    {
        Runtime* rt = current_runtime();
        if (rt == nullptr) { owns_region_ = false; return; }
        Context ctx = current_context();

        is_ = rt->create_index_space(ctx, Rect<1>(0, 0));
        fs_ = rt->create_field_space(ctx);
        {
            FieldAllocator fa = rt->create_field_allocator(ctx, fs_);
            fa.allocate_field(sizeof(T), FID_VAL);
        }
        lr_ = rt->create_logical_region(ctx, is_, fs_);
        owns_region_ = true;

        sync_to_region();
    }

    void destroy_legion_region()
    {
        if (!owns_region_) return;
        Runtime* rt = current_runtime();
        if (rt == nullptr) return;
        Context ctx = current_context();
        rt->destroy_logical_region(ctx, lr_);
        rt->destroy_field_space(ctx, fs_);
        rt->destroy_index_space(ctx, is_);
        owns_region_ = false;
    }

    void sync_to_region()
    {
        if (!owns_region_ || current_runtime() == nullptr) return;
        Runtime* rt = current_runtime();
        Context ctx = current_context();

        InlineLauncher il(
            RegionRequirement(lr_, WRITE_DISCARD, EXCLUSIVE, lr_));
        il.add_field(FID_VAL);
        PhysicalRegion pr = rt->map_region(ctx, il);
        pr.wait_until_valid();
        {
            const FieldAccessor<WRITE_DISCARD, T, 1, coord_t,
                Realm::AffineAccessor<T, 1, coord_t>> acc(pr, FID_VAL);
            acc[0] = data_;
        }
        rt->unmap_region(ctx, pr);
    }

  public:
    future_type queue;

    shared_var()
      : data_()
      , mtx_()
      , lr_(LogicalRegion::NO_REGION)
      , is_(IndexSpace::NO_SPACE)
      , fs_(FieldSpace::NO_SPACE)
      , owns_region_(false)
      , queue()
    {
        create_legion_region();
    }

    shared_var(T const& t)
      : data_(t)
      , mtx_()
      , lr_(LogicalRegion::NO_REGION)
      , is_(IndexSpace::NO_SPACE)
      , fs_(FieldSpace::NO_SPACE)
      , owns_region_(false)
      , queue()
    {
        create_legion_region();
    }

    shared_var(T&& t)
      : data_(std::move(t))
      , mtx_()
      , lr_(LogicalRegion::NO_REGION)
      , is_(IndexSpace::NO_SPACE)
      , fs_(FieldSpace::NO_SPACE)
      , owns_region_(false)
      , queue()
    {
        create_legion_region();
    }

    // Copy constructor – used only for STM snapshots; no region created.
    shared_var(shared_var const& rhs)
      : data_(rhs.data_)
      , mtx_()
      , lr_(LogicalRegion::NO_REGION)
      , is_(IndexSpace::NO_SPACE)
      , fs_(FieldSpace::NO_SPACE)
      , owns_region_(false)
      , queue()
    {}

    ~shared_var() { destroy_legion_region(); }

    // Locks, then clones.  The clone is a lightweight snapshot (no region).
    shared_var_base* clone() const
    {
        auto l = lock();
        return new shared_var(data_, snapshot_tag{});
    }

    // Doesn't lock.
    T const& read() const
    {
        return data_;
    }

    // Doesn't lock.
    void write(T const& rhs)
    {
        data_ = rhs;
    }

    // Doesn't lock.
    void write(shared_var_base const& rhs)
    {
        data_ = dynamic_cast<shared_var const*>(&rhs)->read();
    }

    std::unique_lock<std::mutex> lock() const
    {
        return std::unique_lock<std::mutex>(mtx_);
    }

    bool operator==(shared_var_base const& rhs) const
    {
        return data_ == dynamic_cast<shared_var const*>(&rhs)->read();
    }

    local_var get_local(transaction& trans)
    {
        return local_var(&trans, this);
    }

    // Expose the backing Legion region so callers can build
    // RegionRequirements for child-task launches.
    LogicalRegion get_logical_region() const { return lr_; }

    // Push the current in-memory value into the Legion region.
    void flush_to_region() { sync_to_region(); }
};

// ---------------------------------------------------------------------------
// transaction  –  optimistic, compare-and-swap style STM
//
// Algorithm on commit:
//   1) Lock all touched variables (sorted pointer order → deadlock-free).
//   2) Validate recorded reads against current values.
//   3) Perform buffered writes.
//   4) Fire registered async continuations.
//   5) Release locks (RAII).
// ---------------------------------------------------------------------------
struct transaction
{
    std::list<
        std::pair<
            shared_var_base*                     // shared variable read
          , std::shared_ptr<shared_var_base>     // snapshot at read-time
        >
    > read_list;

    std::set<
        shared_var_base*                         // shared variable written
    > write_set;

    std::list<
        std::pair<
            astm_future*                         // future to chain onto (NULL ⇒ fire-and-forget)
          , std::function<void(transaction*)>    // continuation
        >
    > async_list;

    std::map<
        shared_var_base*                         // shared variable
      , std::shared_ptr<shared_var_base>         // transaction-local value
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
        // 1.) Obtain exclusive access (sorted map → deterministic lock order).
        std::list<std::unique_lock<std::mutex>> locks;

        for ( std::pair<shared_var_base*, std::shared_ptr<shared_var_base>> const& var
            : variables)
        {
            assert(var.first != NULL);
            locks.push_back((*var.first).lock());
        }

        // 2.) Validate reads.
        for ( std::pair<shared_var_base*, std::shared_ptr<shared_var_base>> const& var
            : read_list)
        {
            assert(var.first != NULL);

            if (!((*var.first) == (*var.second)))
            {
                clear();
                return false;               // conflict detected – caller retries
            }
        }

        // 3.) Perform writes.
        for (shared_var_base* var : write_set)
        {
            assert(var != NULL);

            auto it = variables.find(var);
            assert(it != variables.end());

            (*var).write((*(*it).second));
        }

        // 4.) Fire async continuations.
        for ( std::pair<astm_future*, std::function<void(transaction*)>>& op
            : async_list)
        {
            if (op.first == NULL)
                // Fire-and-forget.
                std::async(std::launch::async, op.second, this);
            else
                (*op.first) = (*op.first).then(std::bind(op.second, this));
        }

        // 5.) Locks released by RAII.
        return true;
    }

    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != NULL);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base>> entry(var, nullptr);

        auto result = variables.insert(entry);

        if (result.second == true)
        {
            // First read – snapshot the current value.
            (*result.first).second.reset((*var).clone());

            read_list.push_back(*result.first);

            return (*(*result.first).second);
        }
        else
            return (*(*result.first).second);
    }

    void write(shared_var_base* var, shared_var_base const& value)
    {
        assert(var != NULL);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base>>
            entry(var, std::shared_ptr<shared_var_base>(value.clone()));

        auto result = variables.insert(entry);

        if (result.second == true)
        {
            write_set.insert(var);
        }
        else
        {
            (*result.first).second = entry.second;
            write_set.insert(var);
        }
    }

    void then(astm_future* fut, std::function<void(transaction*)> F)
    {
        std::pair<astm_future*, std::function<void(transaction*)>> entry(fut, F);
        async_list.push_back(entry);
    }
};

// ---------------------------------------------------------------------------
// local_var member implementations (need full transaction definition)
// ---------------------------------------------------------------------------

template <typename T>
shared_var<T>::local_var::operator T const& () const
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
    shared_var tmp(rhs, snapshot_tag{});
    trans_->write(var_, tmp);
    return *this;
}

template <typename T>
template <typename F>
void shared_var<T>::local_var::then(F f)
{
    assert(trans_);
    trans_->then(&dynamic_cast<shared_var*>(var_)->queue, f);
}

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_);
    trans_->then(&fut_, f);
}

} // namespace astm

// ---------------------------------------------------------------------------
// Test / report macros (used by the unit-test and benchmark .cpp files)
// ---------------------------------------------------------------------------
#include <cassert>
#define ASTM_TEST  assert
#define ASTM_REPORT 0

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
