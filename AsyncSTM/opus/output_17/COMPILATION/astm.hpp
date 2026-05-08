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
#include <fstream>

using namespace Legion;

// Macros for testing (replacing astm_config.hpp definitions)
#ifndef ASTM_TEST
#define ASTM_TEST assert
#endif

#ifndef ASTM_REPORT
#define ASTM_REPORT 0
#endif

namespace astm
{

// ----------------------------------------------------------------
// Field ID for data stored in logical regions
// ----------------------------------------------------------------
enum { FID_VAR_DATA = 101 };

// ----------------------------------------------------------------
// Thread-local Legion runtime / context – must be set at the
// beginning of every Legion task via set_legion_context().
// ----------------------------------------------------------------
inline thread_local Runtime* legion_runtime = nullptr;
inline thread_local Context  legion_ctx;

inline void set_legion_context(Runtime* rt, Context ctx)
{
    legion_runtime = rt;
    legion_ctx     = ctx;
}

// ----------------------------------------------------------------
// RAII guard that holds a mapped PhysicalRegion and unmaps it on
// destruction.  Replaces std::unique_lock<std::mutex>.
// ----------------------------------------------------------------
struct region_guard
{
    Runtime*       rt_;
    Context        ctx_;
    PhysicalRegion pr_;
    bool           active_;

    region_guard()
      : rt_(nullptr), active_(false)
    {}

    region_guard(Runtime* rt, Context ctx, PhysicalRegion pr)
      : rt_(rt), ctx_(ctx), pr_(pr), active_(true)
    {}

    region_guard(region_guard&& o) noexcept
      : rt_(o.rt_), ctx_(o.ctx_), pr_(o.pr_), active_(o.active_)
    {
        o.active_ = false;
    }

    region_guard& operator=(region_guard&& o) noexcept
    {
        if (this != &o)
        {
            if (active_ && rt_) rt_->unmap_region(ctx_, pr_);
            rt_     = o.rt_;
            ctx_    = o.ctx_;
            pr_     = o.pr_;
            active_ = o.active_;
            o.active_ = false;
        }
        return *this;
    }

    ~region_guard()
    {
        if (active_ && rt_) rt_->unmap_region(ctx_, pr_);
    }

    region_guard(region_guard const&)            = delete;
    region_guard& operator=(region_guard const&) = delete;

    PhysicalRegion get() const { return pr_; }
};

// ----------------------------------------------------------------
// Lightweight synchronous future.  All callbacks execute
// immediately (within the calling Legion task).  Replaces
// hpx::future<void> / std::future<void>.
// ----------------------------------------------------------------
struct simple_future
{
    simple_future() {}

    simple_future then(std::function<void()> f)
    {
        f();
        return simple_future();
    }

    void get() { /* always ready */ }
};

inline simple_future make_ready_future() { return simple_future(); }

// ----------------------------------------------------------------
// shared_var_base – type-erased interface for transactional vars
// ----------------------------------------------------------------
struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;

    virtual void write(shared_var_base const&) = 0;

    // Map the region with READ_WRITE/EXCLUSIVE and return an RAII guard
    virtual region_guard lock() const = 0;

    virtual bool operator==(shared_var_base const&) const = 0;

    // Compare the value inside an *already-mapped* PhysicalRegion with
    // the value held in a snapshot shared_var_base.
    virtual bool compare_mapped(PhysicalRegion pr,
                                shared_var_base const& snapshot) const = 0;

    // Write the snapshot's value into an *already-mapped* PhysicalRegion.
    virtual void write_mapped(PhysicalRegion pr,
                              shared_var_base const& source) = 0;
};

struct transaction;

// ----------------------------------------------------------------
// transaction_future – thin wrapper used for .then() chaining
// ----------------------------------------------------------------
struct transaction_future
{
    typedef simple_future future_type;

  private:
    transaction* trans_;
    future_type  fut_;

  public:
    transaction_future(transaction* trans)
      : trans_(trans), fut_()
    {}

    transaction_future(transaction& trans)
      : trans_(&trans), fut_()
    {}

    template <typename F>
    void then(F f);

    void get()
    {
        fut_.get();
    }
};

// ----------------------------------------------------------------
// shared_var<T> – a single value of type T backed by a
// one-element LogicalRegion.  Snapshot copies (produced by clone)
// store the value in-memory only.
// ----------------------------------------------------------------
template <typename T>
struct shared_var : shared_var_base
{
    typedef simple_future future_type;

    // ---------- local_var (transactional proxy) ----------
    struct local_var
    {
      private:
        transaction*    trans_;
        shared_var_base* var_;

      public:
        local_var(transaction* trans, shared_var_base* var)
          : trans_(trans), var_(var)
        {}

        T get() const;

        operator T const& () const;

        local_var& operator=(shared_var_base const& rhs);

        local_var& operator=(T const& rhs);

        template <typename F>
        void then(F f);
    };

  private:
    // Tag type so the snapshot constructor is unambiguous
    struct snapshot_tag {};

    T              data_;         // cached / snapshot value
    bool           is_snapshot_;  // true  => data_ is authoritative
                                  // false => LogicalRegion is authoritative
    LogicalRegion  region_;
    IndexSpace     is_;
    FieldSpace     fs_;

    // ------- internal helpers for region I/O -------

    void create_region()
    {
        Runtime* rt  = legion_runtime;
        Context  ctx = legion_ctx;
        assert(rt != nullptr);

        is_ = rt->create_index_space(ctx, Rect<1>(0, 0));
        fs_ = rt->create_field_space(ctx);
        {
            FieldAllocator fa = rt->create_field_allocator(ctx, fs_);
            fa.allocate_field(sizeof(T), FID_VAR_DATA);
        }
        region_ = rt->create_logical_region(ctx, is_, fs_);
    }

    void write_to_region(T const& val) const
    {
        Runtime* rt  = legion_runtime;
        Context  ctx = legion_ctx;

        InlineLauncher il(
            RegionRequirement(region_, WRITE_DISCARD, EXCLUSIVE, region_));
        il.add_field(FID_VAR_DATA);
        PhysicalRegion pr = rt->map_region(ctx, il);
        pr.wait_until_valid();

        const FieldAccessor<WRITE_DISCARD, T, 1, coord_t,
            Realm::AffineAccessor<T, 1, coord_t>> acc(pr, FID_VAR_DATA);
        acc[0] = val;

        rt->unmap_region(ctx, pr);
    }

    T read_from_region() const
    {
        Runtime* rt  = legion_runtime;
        Context  ctx = legion_ctx;

        InlineLauncher il(
            RegionRequirement(region_, READ_ONLY, EXCLUSIVE, region_));
        il.add_field(FID_VAR_DATA);
        PhysicalRegion pr = rt->map_region(ctx, il);
        pr.wait_until_valid();

        const FieldAccessor<READ_ONLY, T, 1, coord_t,
            Realm::AffineAccessor<T, 1, coord_t>> acc(pr, FID_VAR_DATA);
        T val = acc[0];

        rt->unmap_region(ctx, pr);
        return val;
    }

    // Snapshot-only constructor (no region created)
    shared_var(T const& t, snapshot_tag)
      : data_(t), is_snapshot_(true), queue()
    {}

  public:
    future_type queue;

    // ---- Region-backed constructors ----

    shared_var()
      : data_(), is_snapshot_(false), queue()
    {
        create_region();
        write_to_region(data_);
    }

    shared_var(T const& t)
      : data_(t), is_snapshot_(false), queue()
    {
        create_region();
        write_to_region(t);
    }

    shared_var(T&& t)
      : data_(std::move(t)), is_snapshot_(false), queue()
    {
        create_region();
        write_to_region(data_);
    }

    shared_var(shared_var const& rhs)
      : data_(), is_snapshot_(false), queue()
    {
        data_ = rhs.is_snapshot_ ? rhs.data_ : rhs.read_from_region();
        create_region();
        write_to_region(data_);
    }

    ~shared_var()
    {
        if (!is_snapshot_ && legion_runtime != nullptr)
        {
            legion_runtime->destroy_logical_region(legion_ctx, region_);
            legion_runtime->destroy_field_space(legion_ctx, fs_);
            legion_runtime->destroy_index_space(legion_ctx, is_);
        }
    }

    // ---- shared_var_base virtual interface ----

    // Produces a lightweight snapshot (no region)
    shared_var_base* clone() const override
    {
        if (is_snapshot_)
            return new shared_var(data_, snapshot_tag{});

        T val = read_from_region();
        return new shared_var(val, snapshot_tag{});
    }

    // Direct read – returns reference to cached data_.
    // For region-backed vars the cache is refreshed from the region.
    T const& read() const
    {
        if (!is_snapshot_)
            const_cast<shared_var*>(this)->data_ = read_from_region();
        return data_;
    }

    // Direct write – bypasses transactions (used e.g. in the retry test).
    void write(T const& rhs)
    {
        if (is_snapshot_)
            data_ = rhs;
        else
            write_to_region(rhs);
    }

    // Virtual write from another shared_var_base (always a snapshot).
    void write(shared_var_base const& rhs) override
    {
        T val = dynamic_cast<shared_var const*>(&rhs)->data_;
        write(val);
    }

    // Map the region exclusively – returns an RAII guard.
    region_guard lock() const override
    {
        assert(!is_snapshot_);
        Runtime* rt  = legion_runtime;
        Context  ctx = legion_ctx;

        InlineLauncher il(
            RegionRequirement(region_, READ_WRITE, EXCLUSIVE, region_));
        il.add_field(FID_VAR_DATA);
        PhysicalRegion pr = rt->map_region(ctx, il);
        pr.wait_until_valid();

        return region_guard(rt, ctx, pr);
    }

    // Equality via snapshot data or region read.
    bool operator==(shared_var_base const& rhs) const override
    {
        T my_val    = is_snapshot_ ? data_ : read_from_region();
        T other_val = dynamic_cast<shared_var const*>(&rhs)->data_;
        return my_val == other_val;
    }

    // Compare the value inside an already-mapped region against a snapshot.
    bool compare_mapped(PhysicalRegion pr,
                        shared_var_base const& snapshot) const override
    {
        const FieldAccessor<READ_WRITE, T, 1, coord_t,
            Realm::AffineAccessor<T, 1, coord_t>> acc(pr, FID_VAR_DATA);
        T current  = acc[0];
        T snap_val = dynamic_cast<shared_var const*>(&snapshot)->data_;
        return current == snap_val;
    }

    // Write snapshot value into an already-mapped region.
    void write_mapped(PhysicalRegion pr,
                      shared_var_base const& source) override
    {
        const FieldAccessor<READ_WRITE, T, 1, coord_t,
            Realm::AffineAccessor<T, 1, coord_t>> acc(pr, FID_VAR_DATA);
        acc[0] = dynamic_cast<shared_var const*>(&source)->data_;
    }

    LogicalRegion get_region() const { return region_; }

    local_var get_local(transaction& trans)
    {
        return local_var(&trans, this);
    }
};

// ----------------------------------------------------------------
// transaction
// ----------------------------------------------------------------
struct transaction
{
    std::list<
        std::pair<
            shared_var_base*                     // shared variable
          , std::shared_ptr<shared_var_base>     // recorded value
        >
    > read_list;

    std::set<
        shared_var_base*                         // variables written
    > write_set;

    std::list<
        std::pair<
            simple_future*                       // future (NULL => fire-and-forget)
          , std::function<void(transaction*)>    // callback
        >
    > async_list;

    std::map<
        shared_var_base*                         // variable
      , std::shared_ptr<shared_var_base>         // current internal value
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
        // 1.) Map every touched region with READ_WRITE / EXCLUSIVE.
        //     The std::map is pointer-ordered, giving a consistent
        //     mapping order (analogous to lock ordering).
        std::map<shared_var_base*, region_guard> guards;

        for (auto& var : variables)
        {
            assert(var.first != NULL);
            guards[var.first] = var.first->lock();
        }

        // 2.) Verify recorded reads against the current region values.
        for (auto& var : read_list)
        {
            assert(var.first != NULL);

            auto git = guards.find(var.first);
            assert(git != guards.end());

            if (!var.first->compare_mapped(git->second.get(), *var.second))
            {
                clear();
                guards.clear();          // RAII unmaps all regions
                return false;
            }
        }

        // 3.) Perform writes through the mapped regions.
        for (shared_var_base* var : write_set)
        {
            assert(var != NULL);

            auto vit = variables.find(var);
            assert(vit != variables.end());

            auto git = guards.find(var);
            assert(git != guards.end());

            var->write_mapped(git->second.get(), *vit->second);
        }

        // 4.) Execute async / then callbacks (synchronously, within
        //     the current Legion task).
        for (auto& op : async_list)
        {
            if (op.first == NULL)
                op.second(this);                           // fire-and-forget
            else
                *op.first = op.first->then(
                    std::bind(op.second, this));
        }

        // 5.) Release exclusive access (RAII on guards going out of scope).
        return true;
    }

    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != NULL);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base>>
            entry(var, nullptr);

        auto result = variables.insert(entry);

        if (result.second == true)
        {
            // First access – clone the region's current value.
            (*result.first).second.reset((*var).clone());

            // Record the read.
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

    void then(simple_future* fut,
              std::function<void(transaction*)> F)
    {
        std::pair<simple_future*, std::function<void(transaction*)>>
            entry(fut, F);
        async_list.push_back(entry);
    }
};

// ----------------------------------------------------------------
// shared_var<T>::local_var method definitions
// ----------------------------------------------------------------

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
    shared_var tmp(rhs);
    trans_->write(var_, tmp);
    return *this;
}

template <typename T>
template <typename F>
void shared_var<T>::local_var::then(F f)
{
    assert(trans_);
    trans_->then(
        &dynamic_cast<shared_var*>(var_)->queue, f);
}

// ----------------------------------------------------------------
// transaction_future::then
// ----------------------------------------------------------------

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_);
    trans_->then(&fut_, f);
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
