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

// -------------------------------------------------------------------------
// Legion context management – must be called at the start of every task
// that uses ASTM primitives.
// -------------------------------------------------------------------------
enum AstmFieldIDs { FID_VAL = 200 };

inline thread_local Runtime* tl_runtime = nullptr;
inline thread_local Context  tl_context;

inline void set_legion_context(Runtime* rt, Context ctx)
{
    tl_runtime = rt;
    tl_context = ctx;
}

inline Runtime* get_runtime() { return tl_runtime; }
inline Context  get_context() { return tl_context; }

// -------------------------------------------------------------------------
// shared_var_base – type-erased interface for transactional variables
// -------------------------------------------------------------------------
struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;

    virtual void write(shared_var_base const&) = 0;

    virtual bool operator==(shared_var_base const&) const = 0;

    // Legion helpers
    virtual LogicalRegion get_region() const = 0;
    virtual bool          has_region() const = 0;
};

struct transaction;

// -------------------------------------------------------------------------
// transaction_future – thin wrapper; async actions are executed
// synchronously at commit time in the Legion translation.
// -------------------------------------------------------------------------
struct transaction_future
{
  private:
    transaction* trans_;

  public:
    transaction_future(transaction* trans)
      : trans_(trans)
    {}

    transaction_future(transaction& trans)
      : trans_(&trans)
    {}

    template <typename F>
    void then(F f);

    void get() { /* synchronous in Legion – nothing to wait on */ }
};

// -------------------------------------------------------------------------
// shared_var<T>
// -------------------------------------------------------------------------
template <typename T>
struct shared_var : shared_var_base
{
    // ----- local_var (transaction-scoped proxy) --------------------------
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
    // Tag type used to create lightweight "snapshot" copies that are NOT
    // backed by a LogicalRegion.
    struct snapshot_tag {};

    mutable T      data_;
    LogicalRegion  region_;
    IndexSpace     is_;
    FieldSpace     fs_;
    bool           owns_region_;

    // ---- Region helpers ------------------------------------------------
    void create_region()
    {
        assert(tl_runtime != nullptr);
        Rect<1> bounds(0, 0);
        is_ = tl_runtime->create_index_space(tl_context, bounds);
        fs_ = tl_runtime->create_field_space(tl_context);
        {
            FieldAllocator fa =
                tl_runtime->create_field_allocator(tl_context, fs_);
            fa.allocate_field(sizeof(T), FID_VAL);
        }
        region_ = tl_runtime->create_logical_region(tl_context, is_, fs_);
        owns_region_ = true;
    }

    void write_region_value(T const& val) const
    {
        assert(owns_region_ && tl_runtime != nullptr);
        InlineLauncher il(
            RegionRequirement(region_, WRITE_DISCARD, EXCLUSIVE, region_));
        il.add_field(FID_VAL);
        PhysicalRegion pr = tl_runtime->map_region(tl_context, il);
        pr.wait_until_valid();
        const FieldAccessor<WRITE_DISCARD, T, 1, coord_t,
            Realm::AffineAccessor<T, 1, coord_t>> acc(pr, FID_VAL);
        acc[0] = val;
        tl_runtime->unmap_region(tl_context, pr);
    }

    T read_region_value() const
    {
        assert(owns_region_ && tl_runtime != nullptr);
        InlineLauncher il(
            RegionRequirement(region_, READ_ONLY, EXCLUSIVE, region_));
        il.add_field(FID_VAL);
        PhysicalRegion pr = tl_runtime->map_region(tl_context, il);
        pr.wait_until_valid();
        const FieldAccessor<READ_ONLY, T, 1, coord_t,
            Realm::AffineAccessor<T, 1, coord_t>> acc(pr, FID_VAL);
        T val = acc[0];
        tl_runtime->unmap_region(tl_context, pr);
        return val;
    }

    // Snapshot constructor – no region allocated.
    shared_var(T const& t, snapshot_tag)
      : data_(t)
      , region_(LogicalRegion::NO_REGION)
      , is_(IndexSpace::NO_SPACE)
      , fs_(FieldSpace::NO_SPACE)
      , owns_region_(false)
    {}

  public:
    // ---- Public constructors – all create a backing LogicalRegion -------
    shared_var()
      : data_()
      , region_(LogicalRegion::NO_REGION)
      , is_(IndexSpace::NO_SPACE)
      , fs_(FieldSpace::NO_SPACE)
      , owns_region_(false)
    {
        create_region();
        write_region_value(T());
    }

    shared_var(T const& t)
      : data_(t)
      , region_(LogicalRegion::NO_REGION)
      , is_(IndexSpace::NO_SPACE)
      , fs_(FieldSpace::NO_SPACE)
      , owns_region_(false)
    {
        create_region();
        write_region_value(t);
    }

    shared_var(T&& t)
      : data_(std::move(t))
      , region_(LogicalRegion::NO_REGION)
      , is_(IndexSpace::NO_SPACE)
      , fs_(FieldSpace::NO_SPACE)
      , owns_region_(false)
    {
        create_region();
        write_region_value(data_);
    }

    shared_var(shared_var const& rhs)
      : data_(rhs.data_)
      , region_(LogicalRegion::NO_REGION)
      , is_(IndexSpace::NO_SPACE)
      , fs_(FieldSpace::NO_SPACE)
      , owns_region_(false)
    {
        create_region();
        write_region_value(data_);
    }

    ~shared_var()
    {
        if (owns_region_ && tl_runtime != nullptr)
        {
            tl_runtime->destroy_logical_region(tl_context, region_);
            tl_runtime->destroy_field_space(tl_context, fs_);
            tl_runtime->destroy_index_space(tl_context, is_);
        }
    }

    // ---- shared_var_base interface -------------------------------------

    // clone() produces a snapshot (region-free) copy.
    // For region-backed vars the current value is read from the region,
    // mirroring the lock-and-copy semantics of the original.
    shared_var_base* clone() const override
    {
        T val = owns_region_ ? read_region_value() : data_;
        return new shared_var(val, snapshot_tag{});
    }

    // Direct read – syncs from region if backed by one.
    T const& read() const
    {
        if (owns_region_)
            data_ = read_region_value();
        return data_;
    }

    // Direct write (possibly outside a transaction).
    void write(T const& rhs)
    {
        data_ = rhs;
        if (owns_region_)
            write_region_value(rhs);
    }

    // Write from another shared_var_base (used by transaction commit).
    void write(shared_var_base const& rhs) override
    {
        data_ = dynamic_cast<shared_var const*>(&rhs)->data_;
        if (owns_region_)
            write_region_value(data_);
    }

    // Equality check: compares the *current* value (reading from region if
    // necessary) against the snapshot's cached data_.
    bool operator==(shared_var_base const& rhs) const override
    {
        T current = owns_region_ ? read_region_value() : data_;
        return current == dynamic_cast<shared_var const*>(&rhs)->data_;
    }

    LogicalRegion get_region() const override { return region_; }
    bool          has_region() const override { return owns_region_; }

    local_var get_local(transaction& trans)
    {
        return local_var(&trans, this);
    }
};

// -------------------------------------------------------------------------
// transaction
// -------------------------------------------------------------------------
struct transaction
{
    std::list<
        std::pair<
            shared_var_base*                     // live variable
          , std::shared_ptr<shared_var_base>      // snapshot at read-time
        >
    > read_list;

    std::set<
        shared_var_base*                          // variables written
    > write_set;

    std::list<
        std::pair<
            void*                                 // unused (was future ptr)
          , std::function<void(transaction*)>      // callback
        >
    > async_list;

    std::map<
        shared_var_base*                          // live variable
      , std::shared_ptr<shared_var_base>          // current buffered value
    > variables;

    void clear()
    {
        read_list.clear();
        write_set.clear();
        async_list.clear();
        variables.clear();
    }

    // Commit algorithm (Legion version):
    //
    // Within a single Legion task execution is sequential, so we do NOT
    // need to acquire any locks.  Cross-task atomicity is guaranteed by
    // the Legion runtime through region privileges.
    //
    // 1. Validate recorded reads against current region values.
    // 2. Perform buffered writes to the live variables (and their regions).
    // 3. Execute deferred callbacks synchronously.
    bool commit_transaction()
    {
        // 1. Validate reads.
        for ( std::pair<shared_var_base*, std::shared_ptr<shared_var_base>> const& var
            : read_list)
        {
            assert(var.first != nullptr);

            if (!((*var.first) == (*var.second)))
            {
                clear();
                return false;
            }
        }

        // 2. Perform writes – the virtual write() on the live variable
        //    pushes the value both into data_ and into the LogicalRegion.
        for (shared_var_base* var : write_set)
        {
            assert(var != nullptr);

            auto it = variables.find(var);
            assert(it != variables.end());

            (*var).write((*(*it).second));
        }

        // 3. Execute deferred async callbacks synchronously.
        for ( auto& op : async_list)
        {
            op.second(this);
        }

        return true;
    }

    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != nullptr);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base>>
            entry(var, nullptr);

        auto result = variables.insert(entry);

        if (result.second == true)
        {
            // First access – snapshot the current value from the region.
            (*result.first).second.reset((*var).clone());

            // Record in read list (shared_ptr is copied so the snapshot
            // survives even if a later write replaces it in the map).
            read_list.push_back(*result.first);

            return (*(*result.first).second);
        }
        else
        {
            return (*(*result.first).second);
        }
    }

    void write(shared_var_base* var, shared_var_base const& value)
    {
        assert(var != nullptr);

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

    void then(void* /*fut*/, std::function<void(transaction*)> F)
    {
        async_list.push_back({nullptr, F});
    }
};

// -------------------------------------------------------------------------
// local_var out-of-line definitions
// -------------------------------------------------------------------------
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
    // Build a lightweight snapshot (no LogicalRegion) to avoid the cost
    // of region creation for a value that is only used as a buffer.
    shared_var tmp(rhs, snapshot_tag{});
    trans_->write(var_, tmp);
    return *this;
}

template <typename T>
template <typename F>
void shared_var<T>::local_var::then(F f)
{
    assert(trans_);
    trans_->then(nullptr, f);
}

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_);
    trans_->then(nullptr, f);
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
