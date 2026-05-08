////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Translated to the Legion execution model.
////////////////////////////////////////////////////////////////////////////////

#if !defined(ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94)
#define ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94

#include "astm_config.hpp"

#include <legion.h>

#include <list>
#include <set>
#include <map>
#include <vector>
#include <functional>
#include <memory>
#include <cassert>

using namespace Legion;

namespace astm
{

// ============================================================================
// Thread-local Legion runtime and context.
// Must be set at the start of every Legion task (including the top-level task).
// ============================================================================
inline thread_local Runtime* legion_runtime = nullptr;
inline thread_local Context  legion_ctx;

inline void set_legion_context(Runtime* rt, Context ctx)
{
    legion_runtime = rt;
    legion_ctx     = ctx;
}

// Field ID used for the single value stored in every shared_var region.
enum { FID_SHARED_VAR_DATA = 200 };

// ============================================================================
// Simple future replacement.
// Legion Future<void> does not support .then(); we implement a lightweight
// synchronous callback chain instead.  Callbacks are accumulated and executed
// in order when get() is called (or when chained during commit).
// ============================================================================
struct legion_future
{
    std::vector<std::function<void()>> pending_;

    legion_future()                                = default;
    legion_future(legion_future const&)            = default;
    legion_future(legion_future&&)                 = default;
    legion_future& operator=(legion_future const&) = default;
    legion_future& operator=(legion_future&&)      = default;

    template <typename F>
    legion_future then(F f)
    {
        legion_future result;
        result.pending_ = pending_;
        result.pending_.push_back(std::function<void()>([f]() { f(); }));
        return result;
    }

    void get()
    {
        for (auto& f : pending_) f();
        pending_.clear();
    }
};

inline legion_future make_legion_ready_future()
{
    return legion_future();
}

// ============================================================================
// Dummy lock.
// In Legion the runtime manages coherence; explicit locking is unnecessary.
// This type exists only so that the commit_transaction() interface compiles
// unchanged (it collects "locks" in a list via RAII).
// ============================================================================
struct legion_lock
{
    legion_lock()                              = default;
    legion_lock(legion_lock&&)                 = default;
    legion_lock& operator=(legion_lock&&)      = default;
    legion_lock(legion_lock const&)            = delete;
    legion_lock& operator=(legion_lock const&) = delete;
};

// ============================================================================
// shared_var_base – abstract interface for all shared variables.
// ============================================================================
struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;

    virtual void write(shared_var_base const&) = 0;

    virtual legion_lock lock() const { return legion_lock(); }

    virtual bool operator==(shared_var_base const&) const = 0;
};

struct transaction;

// ============================================================================
// transaction_future – a future bound to a specific transaction.
// ============================================================================
struct transaction_future
{
    typedef legion_future future_type;

  private:
    transaction* trans_;
    future_type  fut_;

  public:
    transaction_future(transaction* trans)
      : trans_(trans)
      , fut_(make_legion_ready_future())
    {}

    transaction_future(transaction& trans)
      : trans_(&trans)
      , fut_(make_legion_ready_future())
    {}

    template <typename F>
    void then(F f);

    void get()
    {
        fut_.get();
    }
};

// ============================================================================
// shared_var<T> – a transactional shared variable backed by a Legion
//                 LogicalRegion (one element, one field of type T).
//
// "Primary" instances own a region; "clones" (produced by clone()) carry only
// a cached value and are used inside the transaction bookkeeping.
// ============================================================================
template <typename T>
struct shared_var : shared_var_base
{
    typedef legion_future future_type;

    // ---- local_var: transaction-local proxy for a shared variable ---------
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
    // ---- data and region bookkeeping -------------------------------------
    mutable T           data_;
    LogicalRegion       lr_;
    IndexSpaceT<1>      is_;
    FieldSpace          fs_;
    bool                has_region_;

    // Tag type so the clone constructor is distinguishable.
    struct clone_tag_t {};

    // -- Region helpers ---------------------------------------------------

    void create_region()
    {
        if (legion_runtime == nullptr) { has_region_ = false; return; }
        Runtime* rt  = legion_runtime;
        Context  ctx = legion_ctx;
        is_ = rt->create_index_space(ctx, Rect<1>(0, 0));
        fs_ = rt->create_field_space(ctx);
        {
            FieldAllocator fa = rt->create_field_allocator(ctx, fs_);
            fa.allocate_field(sizeof(T), FID_SHARED_VAR_DATA);
        }
        lr_         = rt->create_logical_region(ctx, is_, fs_);
        has_region_ = true;
    }

    void destroy_region()
    {
        if (!has_region_ || legion_runtime == nullptr) return;
        Runtime* rt  = legion_runtime;
        Context  ctx = legion_ctx;
        rt->destroy_logical_region(ctx, lr_);
        rt->destroy_field_space(ctx, fs_);
        rt->destroy_index_space(ctx, is_);
        lr_         = LogicalRegion::NO_REGION;
        has_region_ = false;
    }

    // Write cached data_ → region.
    void sync_to_region() const
    {
        if (!has_region_ || legion_runtime == nullptr) return;
        Runtime* rt  = legion_runtime;
        Context  ctx = legion_ctx;
        InlineLauncher il(
            RegionRequirement(lr_, WRITE_DISCARD, EXCLUSIVE, lr_));
        il.add_field(FID_SHARED_VAR_DATA);
        PhysicalRegion pr = rt->map_region(ctx, il);
        pr.wait_until_valid();
        const FieldAccessor<WRITE_DISCARD, T, 1> acc(pr, FID_SHARED_VAR_DATA);
        acc[Point<1>(0)] = data_;
        rt->unmap_region(ctx, pr);
    }

    // Read region → cached data_.
    void sync_from_region() const
    {
        if (!has_region_ || legion_runtime == nullptr) return;
        Runtime* rt  = legion_runtime;
        Context  ctx = legion_ctx;
        InlineLauncher il(
            RegionRequirement(lr_, READ_ONLY, EXCLUSIVE, lr_));
        il.add_field(FID_SHARED_VAR_DATA);
        PhysicalRegion pr = rt->map_region(ctx, il);
        pr.wait_until_valid();
        const FieldAccessor<READ_ONLY, T, 1> acc(pr, FID_SHARED_VAR_DATA);
        const_cast<T&>(data_) = acc[Point<1>(0)];
        rt->unmap_region(ctx, pr);
    }

    // Private constructor for clones (no region, just cached value).
    shared_var(T const& t, clone_tag_t)
      : data_(t)
      , lr_(LogicalRegion::NO_REGION)
      , has_region_(false)
      , queue(make_legion_ready_future())
    {}

  public:
    future_type queue;

    // ---- Constructors / destructor --------------------------------------

    shared_var()
      : data_()
      , has_region_(false)
      , queue(make_legion_ready_future())
    {
        create_region();
        sync_to_region();
    }

    shared_var(T const& t)
      : data_(t)
      , has_region_(false)
      , queue(make_legion_ready_future())
    {
        create_region();
        sync_to_region();
    }

    shared_var(T&& t)
      : data_(std::move(t))
      , has_region_(false)
      , queue(make_legion_ready_future())
    {
        create_region();
        sync_to_region();
    }

    shared_var(shared_var const& rhs)
      : data_(rhs.data_)
      , has_region_(false)
      , queue(make_legion_ready_future())
    {
        create_region();
        sync_to_region();
    }

    ~shared_var() { destroy_region(); }

    // ---- shared_var_base interface --------------------------------------

    // Reads the current value from the region (if any) and returns a
    // light-weight clone (no region) holding a snapshot.
    shared_var_base* clone() const
    {
        if (has_region_) sync_from_region();
        return new shared_var(data_, clone_tag_t{});
    }

    // Returns cached data.  For clones the cached value IS the value;
    // for primary instances the cache is kept in sync with the region
    // by sync_to_region / sync_from_region.
    T const& read() const
    {
        return data_;
    }

    // Direct write that bypasses the transaction.
    void write(T const& rhs)
    {
        data_ = rhs;
        if (has_region_) sync_to_region();
    }

    // Direct write from another shared_var_base.
    void write(shared_var_base const& rhs)
    {
        data_ = dynamic_cast<shared_var const*>(&rhs)->read();
        if (has_region_) sync_to_region();
    }

    // Dummy lock – Legion manages coherence.
    legion_lock lock() const
    {
        return legion_lock();
    }

    // Compare the *current* region value with the other shared_var's cached
    // value.  Used during commit validation.
    bool operator==(shared_var_base const& rhs) const
    {
        if (has_region_) sync_from_region();
        return data_ == dynamic_cast<shared_var const*>(&rhs)->read();
    }

    local_var get_local(transaction& trans)
    {
        return local_var(&trans, this);
    }
};

// ============================================================================
// transaction – the transactional context.
// ============================================================================
struct transaction
{
    std::list<
        std::pair<
            shared_var_base*                      // The shared variable we read
          , std::shared_ptr<shared_var_base>       // Snapshot at read time
        >
    > read_list;

    std::set<
        shared_var_base*                           // Variables we intend to write
    > write_set;

    std::list<
        std::pair<
            legion_future*                         // Future to chain onto (NULL ⇒ fire-and-forget)
          , std::function<void(transaction*)>       // Async action
        >
    > async_list;

    std::map<
        shared_var_base*                           // Variable
      , std::shared_ptr<shared_var_base>            // Transaction-local value
    > variables;

    void clear()
    {
        read_list.clear();
        write_set.clear();
        async_list.clear();
        variables.clear();
    }

    // -----------------------------------------------------------------
    // commit_transaction()
    //
    // 1) Acquire dummy "locks" (interface compatibility).
    // 2) Validate reads: compare current region values with snapshots.
    // 3) Perform writes: flush transaction-local values to the regions.
    // 4) Execute async callbacks.
    // 5) Return success / failure.
    //
    // Under Legion's default mapper, tasks with conflicting region
    // requirements are serialised, so commits typically succeed.
    // The validation step still catches *self*-modifications made
    // inside the transaction loop (useful for the retry test).
    // -----------------------------------------------------------------
    bool commit_transaction()
    {
        // 1) Dummy locks (RAII-released at end of scope).
        std::list<legion_lock> locks;
        for ( std::pair<shared_var_base*, std::shared_ptr<shared_var_base> > const& var
            : variables)
        {
            assert(var.first != NULL);
            locks.push_back((*var.first).lock());
        }

        // 2) Validate reads.
        for ( std::pair<shared_var_base*, std::shared_ptr<shared_var_base> > const& var
            : read_list)
        {
            assert(var.first != NULL);
            if (!((*var.first) == (*var.second)))
            {
                clear();
                return false;
            }
        }

        // 3) Perform writes – flush transaction-local values to regions.
        for (shared_var_base* var : write_set)
        {
            assert(var != NULL);
            auto it = variables.find(var);
            assert(it != variables.end());
            (*var).write((*(*it).second));
        }

        // 4) Async callbacks.
        for ( std::pair<legion_future*, std::function<void(transaction*)> >& op
            : async_list)
        {
            if (op.first == NULL)
                // Fire-and-forget: execute synchronously.
                op.second(this);
            else
                (*op.first) = (*op.first).then(std::bind(op.second, this));
        }

        // 5) Done.
        return true;
    }

    // -----------------------------------------------------------------
    // read() – returns the transaction-local value of *var.
    //
    // First access: clones the current region value; subsequent accesses
    // return the stored copy.
    // -----------------------------------------------------------------
    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != NULL);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base> >
            entry(var, std::shared_ptr<shared_var_base>());

        auto result = variables.insert(entry);

        if (result.second == true)
        {
            // First read: clone the current region value.
            (*result.first).second.reset((*var).clone());

            // Record the read for later validation.
            read_list.push_back(*result.first);

            return (*(*result.first).second);
        }
        else
            return (*(*result.first).second);
    }

    // -----------------------------------------------------------------
    // write() – record a pending write in the transaction-local map.
    // -----------------------------------------------------------------
    void write(shared_var_base* var, shared_var_base const& value)
    {
        assert(var != NULL);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base> >
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

    // -----------------------------------------------------------------
    // then() – enqueue an async callback.
    // -----------------------------------------------------------------
    void then(legion_future* fut, std::function<void(transaction*)> F)
    {
        std::pair<legion_future*, std::function<void(transaction*)> >
            entry(fut, F);
        async_list.push_back(entry);
    }
};

// ============================================================================
// Out-of-line definitions for shared_var<T>::local_var
// ============================================================================

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
    // Create a lightweight clone (no region) to pass to the transaction.
    shared_var tmp(rhs, clone_tag_t{});
    trans_->write(var_, tmp);
    return *this;
}

template <typename T>
template <typename F>
void shared_var<T>::local_var::then(F f)
{
    assert(trans_);
    shared_var<T>* sv = dynamic_cast<shared_var<T>*>(var_);
    assert(sv);
    trans_->then(&sv->queue, f);
}

// ============================================================================
// Out-of-line definition for transaction_future::then
// ============================================================================

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_);
    trans_->then(&fut_, f);
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
