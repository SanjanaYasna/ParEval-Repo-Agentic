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

namespace astm
{

// Field ID used for shared variable data in Legion logical regions
enum { ASTM_FID_VALUE = 200 };

//--------------------------------------------------------------------------
// Thread-local storage for the current Legion Runtime* and Context.
// Must be initialized via set_legion_context() at the start of every
// Legion task that uses ASTM facilities.
//--------------------------------------------------------------------------
inline Legion::Runtime*& tl_runtime()
{
    static thread_local Legion::Runtime* rt = nullptr;
    return rt;
}

inline Legion::Context& tl_context()
{
    static thread_local Legion::Context ctx;
    return ctx;
}

inline void set_legion_context(Legion::Runtime* rt, Legion::Context ctx)
{
    tl_runtime() = rt;
    tl_context() = ctx;
}

//--------------------------------------------------------------------------
// Base class for all shared transactional variables
//--------------------------------------------------------------------------
struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;

    virtual void write(shared_var_base const&) = 0;

    virtual bool operator==(shared_var_base const&) const = 0;

    // Legion-specific interface -----------------------------------------
    virtual Legion::LogicalRegion get_logical_region() const = 0;
    virtual bool has_region() const = 0;

    // Compare the value currently stored in the physical region against
    // a previously-taken snapshot.  Called during commit validation while
    // the region is mapped READ_WRITE / EXCLUSIVE.
    virtual bool validate_with_region(
        Legion::PhysicalRegion pr,
        shared_var_base const& snapshot) const = 0;

    // Write the source value into the physical region.  Called during
    // commit write-back while the region is mapped READ_WRITE / EXCLUSIVE.
    virtual void write_to_region(
        Legion::PhysicalRegion pr,
        shared_var_base const& source) = 0;

    // Refresh the local data_ cache from a mapped physical region.
    virtual void sync_from_region(Legion::PhysicalRegion pr) = 0;
};

struct transaction;

//--------------------------------------------------------------------------
// Transaction future – allows chaining deferred (post-commit) operations
//--------------------------------------------------------------------------
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

    // In the Legion model the deferred operations are executed inline
    // during commit, so get() is a no-op.
    void get() {}
};

//--------------------------------------------------------------------------
// Shared variable backed by a one-element Legion logical region
//--------------------------------------------------------------------------
template <typename T>
struct shared_var : shared_var_base
{
    // ----- local_var proxy ---------------------------------------------------
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
    Legion::LogicalRegion region_;
    Legion::IndexSpace    is_;
    Legion::FieldSpace    fs_;
    bool                  owns_region_;

    // Tag type used to select the private "clone" constructor
    struct clone_tag {};

    // Private constructor for transaction-internal clones (no region).
    shared_var(T const& t, clone_tag)
      : data_(t)
      , region_(Legion::LogicalRegion::NO_REGION)
      , is_(Legion::IndexSpace::NO_SPACE)
      , fs_(Legion::FieldSpace::NO_SPACE)
      , owns_region_(false)
    {}

    // ------ helpers for direct region I/O ----------------------------------
    void create_region(T const& initial_value)
    {
        Legion::Runtime* rt  = tl_runtime();
        Legion::Context  ctx = tl_context();
        assert(rt != nullptr);

        is_ = rt->create_index_space(ctx, Legion::Rect<1>(0, 0));
        fs_ = rt->create_field_space(ctx);
        {
            Legion::FieldAllocator fa =
                rt->create_field_allocator(ctx, fs_);
            fa.allocate_field(sizeof(T), ASTM_FID_VALUE);
        }
        region_ = rt->create_logical_region(ctx, is_, fs_);

        // Inline-map to initialise
        Legion::InlineLauncher il(Legion::RegionRequirement(
            region_, WRITE_DISCARD, EXCLUSIVE, region_));
        il.add_field(ASTM_FID_VALUE);
        Legion::PhysicalRegion pr = rt->map_region(ctx, il);
        pr.wait_until_valid();
        {
            const Legion::FieldAccessor<WRITE_DISCARD, T, 1> acc(
                pr, ASTM_FID_VALUE);
            acc[0] = initial_value;
        }
        rt->unmap_region(ctx, pr);
        owns_region_ = true;
    }

    void write_region_direct(T const& val)
    {
        if (!owns_region_) return;
        Legion::Runtime* rt  = tl_runtime();
        Legion::Context  ctx = tl_context();

        Legion::InlineLauncher il(Legion::RegionRequirement(
            region_, WRITE_DISCARD, EXCLUSIVE, region_));
        il.add_field(ASTM_FID_VALUE);
        Legion::PhysicalRegion pr = rt->map_region(ctx, il);
        pr.wait_until_valid();
        {
            const Legion::FieldAccessor<WRITE_DISCARD, T, 1> acc(
                pr, ASTM_FID_VALUE);
            acc[0] = val;
        }
        rt->unmap_region(ctx, pr);
    }

    T read_region_direct() const
    {
        if (!owns_region_) return data_;
        Legion::Runtime* rt  = tl_runtime();
        Legion::Context  ctx = tl_context();

        Legion::InlineLauncher il(Legion::RegionRequirement(
            region_, READ_ONLY, EXCLUSIVE, region_));
        il.add_field(ASTM_FID_VALUE);
        Legion::PhysicalRegion pr = rt->map_region(ctx, il);
        pr.wait_until_valid();
        T val;
        {
            const Legion::FieldAccessor<READ_ONLY, T, 1> acc(
                pr, ASTM_FID_VALUE);
            val = acc[0];
        }
        rt->unmap_region(ctx, pr);
        return val;
    }

  public:
    // --- public constructors (each creates an owning region) ---------------
    shared_var()
      : data_()
      , region_(Legion::LogicalRegion::NO_REGION)
      , is_(Legion::IndexSpace::NO_SPACE)
      , fs_(Legion::FieldSpace::NO_SPACE)
      , owns_region_(false)
    { create_region(data_); }

    shared_var(T const& t)
      : data_(t)
      , region_(Legion::LogicalRegion::NO_REGION)
      , is_(Legion::IndexSpace::NO_SPACE)
      , fs_(Legion::FieldSpace::NO_SPACE)
      , owns_region_(false)
    { create_region(t); }

    shared_var(T&& t)
      : data_(t)
      , region_(Legion::LogicalRegion::NO_REGION)
      , is_(Legion::IndexSpace::NO_SPACE)
      , fs_(Legion::FieldSpace::NO_SPACE)
      , owns_region_(false)
    { create_region(data_); }

    // Copy constructor: creates a NEW region with the same initial value.
    shared_var(shared_var const& rhs)
      : data_(rhs.data_)
      , region_(Legion::LogicalRegion::NO_REGION)
      , is_(Legion::IndexSpace::NO_SPACE)
      , fs_(Legion::FieldSpace::NO_SPACE)
      , owns_region_(false)
    { create_region(data_); }

    // Move constructor: transfers region ownership.
    shared_var(shared_var&& rhs)
      : data_(std::move(rhs.data_))
      , region_(rhs.region_)
      , is_(rhs.is_)
      , fs_(rhs.fs_)
      , owns_region_(rhs.owns_region_)
    { rhs.owns_region_ = false; }

    ~shared_var()
    {
        if (owns_region_) {
            Legion::Runtime* rt  = tl_runtime();
            Legion::Context  ctx = tl_context();
            if (rt != nullptr) {
                rt->destroy_logical_region(ctx, region_);
                rt->destroy_field_space(ctx, fs_);
                rt->destroy_index_space(ctx, is_);
            }
        }
    }

    // ----- shared_var_base interface ----------------------------------------

    // Reads the committed value from the region and returns a clone (no region).
    shared_var_base* clone() const override
    {
        if (owns_region_) {
            T val = read_region_direct();
            return new shared_var(val, clone_tag{});
        }
        return new shared_var(data_, clone_tag{});
    }

    // Local (non-transactional) read – returns cached data_.
    T const& read() const
    {
        return data_;
    }

    // Direct (non-transactional) write – updates both cache and region.
    void write(T const& rhs)
    {
        data_ = rhs;
        if (owns_region_)
            write_region_direct(rhs);
    }

    // Writes from another shared_var_base (cache + region).
    void write(shared_var_base const& rhs) override
    {
        data_ = dynamic_cast<shared_var const*>(&rhs)->read();
        if (owns_region_)
            write_region_direct(data_);
    }

    bool operator==(shared_var_base const& rhs) const override
    {
        return data_ == dynamic_cast<shared_var const*>(&rhs)->read();
    }

    // ----- Legion-specific interface ----------------------------------------

    Legion::LogicalRegion get_logical_region() const override
    { return region_; }

    bool has_region() const override
    { return owns_region_; }

    bool validate_with_region(
        Legion::PhysicalRegion pr,
        shared_var_base const& snapshot) const override
    {
        const Legion::FieldAccessor<READ_WRITE, T, 1> acc(
            pr, ASTM_FID_VALUE);
        T current_val = acc[0];
        return current_val ==
               dynamic_cast<shared_var const*>(&snapshot)->read();
    }

    void write_to_region(
        Legion::PhysicalRegion pr,
        shared_var_base const& source) override
    {
        const Legion::FieldAccessor<READ_WRITE, T, 1> acc(
            pr, ASTM_FID_VALUE);
        T val = dynamic_cast<shared_var const*>(&source)->read();
        acc[0] = val;
        data_  = val;   // keep local cache in sync
    }

    void sync_from_region(Legion::PhysicalRegion pr) override
    {
        const Legion::FieldAccessor<READ_ONLY, T, 1> acc(
            pr, ASTM_FID_VALUE);
        data_ = acc[0];
    }

    // ----- transactional accessor ------------------------------------------

    local_var get_local(transaction& trans)
    {
        return local_var(&trans, this);
    }
};

//--------------------------------------------------------------------------
// Transaction
//--------------------------------------------------------------------------
struct transaction
{
    std::list<
        std::pair<
            shared_var_base*                       // original shared variable
          , std::shared_ptr<shared_var_base>        // snapshot value at read time
        >
    > read_list;

    std::set<
        shared_var_base*                           // variables written
    > write_set;

    std::list<
        std::pair<
            void*                                  // unused (was future ptr)
          , std::function<void(transaction*)>       // deferred action
        >
    > async_list;

    std::map<
        shared_var_base*                           // variable
      , std::shared_ptr<shared_var_base>            // transaction-local value
    > variables;

    void clear()
    {
        read_list.clear();
        write_set.clear();
        async_list.clear();
        variables.clear();
    }

    // Commit using Legion inline mappings for exclusive region access.
    bool commit_transaction()
    {
        // Algorithm (mirrors the original):
        //
        // 1.) Inline-map all variable regions READ_WRITE / EXCLUSIVE.
        // 2.) Validate recorded reads against current region values.
        // 3.) Perform write-back from transaction-local state to regions.
        // 4.) Execute deferred (async) operations inline.
        // 5.) Unmap all regions (release exclusive access).

        Legion::Runtime* rt  = tl_runtime();
        Legion::Context  ctx = tl_context();
        assert(rt != nullptr);

        // 1.) Obtain exclusive access via inline mappings (sorted by pointer
        //     address through std::map, giving a deterministic lock order).
        std::map<shared_var_base*, Legion::PhysicalRegion> mapped;

        for (auto const& var : variables)
        {
            assert(var.first != nullptr);
            if (var.first->has_region())
            {
                Legion::LogicalRegion lr = var.first->get_logical_region();
                Legion::InlineLauncher il(Legion::RegionRequirement(
                    lr, READ_WRITE, EXCLUSIVE, lr));
                il.add_field(ASTM_FID_VALUE);
                Legion::PhysicalRegion pr = rt->map_region(ctx, il);
                pr.wait_until_valid();
                mapped[var.first] = pr;
            }
        }

        // 2.) Validate reads.
        for (auto const& var : read_list)
        {
            assert(var.first != nullptr);
            auto mit = mapped.find(var.first);
            if (mit != mapped.end())
            {
                if (!var.first->validate_with_region(
                        mit->second, *var.second))
                {
                    // Transaction fails – unmap and abort.
                    for (auto& m : mapped)
                        rt->unmap_region(ctx, m.second);
                    clear();
                    return false;
                }
            }
        }

        // 3.) Perform writes.
        for (shared_var_base* var : write_set)
        {
            assert(var != nullptr);

            auto vit = variables.find(var);
            assert(vit != variables.end());

            auto mit = mapped.find(var);
            if (mit != mapped.end())
                var->write_to_region(mit->second, *(vit->second));
        }

        // 4.) Execute deferred operations (inline in the Legion model).
        for (auto& op : async_list)
        {
            op.second(this);
        }

        // 5.) Unmap all regions.
        for (auto& m : mapped)
            rt->unmap_region(ctx, m.second);

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
            // First access – snapshot (clone) from the region.
            result.first->second.reset(var->clone());

            // Record the read.
            read_list.push_back(*result.first);

            return *(result.first->second);
        }
        else
        {
            // Already present in transaction-local state.
            return *(result.first->second);
        }
    }

    void write(shared_var_base* var, shared_var_base const& value)
    {
        assert(var != nullptr);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base>>
            entry(var,
                  std::shared_ptr<shared_var_base>(value.clone()));

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

    // Register a deferred (post-commit) operation.
    void then(void* /*fut*/, std::function<void(transaction*)> F)
    {
        async_list.push_back({nullptr, std::move(F)});
    }
};

//--------------------------------------------------------------------------
// Out-of-line template member definitions
//--------------------------------------------------------------------------

template <typename T>
shared_var<T>::local_var::operator T const& () const
{
    return dynamic_cast<shared_var const*>(
        &trans_->read(var_))->read();
}

template <typename T>
T shared_var<T>::local_var::get() const
{
    return dynamic_cast<shared_var const*>(
        &trans_->read(var_))->read();
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
    shared_var tmp(rhs, typename shared_var::clone_tag{});
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
    trans_->then(nullptr, std::function<void(transaction*)>(f));
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
