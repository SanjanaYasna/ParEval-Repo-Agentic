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

// Field ID used for all shared variable regions
enum { FID_VAL = 101 };

// ---------------------------------------------------------------------------
// Thread-local Legion Runtime* and Context management.
// Must be initialised at the start of every Legion task that uses ASTM.
// ---------------------------------------------------------------------------
namespace detail {
    inline Runtime*& runtime_ref() {
        static thread_local Runtime* rt = nullptr;
        return rt;
    }
    inline Context& context_ref() {
        static thread_local Context ctx;
        return ctx;
    }
}

inline void init_legion(Runtime* rt, Context ctx) {
    detail::runtime_ref() = rt;
    detail::context_ref() = ctx;
}
inline Runtime* legion_runtime() { return detail::runtime_ref(); }
inline Context  legion_context() { return detail::context_ref(); }

// ---------------------------------------------------------------------------
// RAII guard that inline-maps a LogicalRegion with READ_WRITE / EXCLUSIVE.
// Acts as the Legion equivalent of std::unique_lock<std::mutex>.
// ---------------------------------------------------------------------------
struct region_lock
{
    PhysicalRegion pr;
    bool active;

    region_lock() : active(false) {}

    explicit region_lock(LogicalRegion lr) : active(true)
    {
        RegionRequirement req(lr, READ_WRITE, EXCLUSIVE, lr);
        req.add_field(FID_VAL);
        InlineLauncher il(req);
        pr = legion_runtime()->map_region(legion_context(), il);
        pr.wait_until_valid();
    }

    region_lock(region_lock&& o) noexcept : pr(o.pr), active(o.active)
    { o.active = false; }

    region_lock& operator=(region_lock&& o) noexcept {
        if (this != &o) { release(); pr = o.pr; active = o.active; o.active = false; }
        return *this;
    }

    void release() {
        if (active) {
            legion_runtime()->unmap_region(legion_context(), pr);
            active = false;
        }
    }

    ~region_lock() { release(); }

    region_lock(const region_lock&) = delete;
    region_lock& operator=(const region_lock&) = delete;
};

// ---------------------------------------------------------------------------
// Base class for all shared variables (type-erased interface).
// ---------------------------------------------------------------------------
struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;

    virtual void write(shared_var_base const&) = 0;

    // Inline-map the backing region with exclusive access (analogous to mutex lock)
    virtual region_lock lock() const = 0;

    virtual bool operator==(shared_var_base const&) const = 0;

    // Legion-specific helpers used during commit
    virtual bool has_region() const = 0;
    virtual LogicalRegion get_region() const = 0;

    // Compare current region value with another shared_var_base (through a
    // PhysicalRegion that is already mapped).
    virtual bool equals_with_pr(PhysicalRegion const& pr,
                                shared_var_base const& other) const = 0;

    // Write value of 'other' into the already-mapped PhysicalRegion.
    virtual void write_with_pr(PhysicalRegion const& pr,
                               shared_var_base const& other) = 0;
};

struct transaction;

// ---------------------------------------------------------------------------
// A lightweight future-like object that stores callbacks to be executed
// synchronously during transaction commit.  get() is a no-op because all
// callbacks have already run by the time commit_transaction() returns true.
// ---------------------------------------------------------------------------
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

    void get()
    {
        // Callbacks were executed synchronously during commit_transaction().
    }
};

// ---------------------------------------------------------------------------
// shared_var<T>  –  a transactional variable backed by a one-element
//                   Legion LogicalRegion.
// ---------------------------------------------------------------------------
template <typename T>
struct shared_var : shared_var_base
{
    // ----- local_var: transaction-local proxy for reads and writes ---------
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
    mutable T data_;          // cached / clone value
    LogicalRegion region_;
    IndexSpace     is_;
    FieldSpace     fs_;
    bool           owns_region_;   // true for "real" variables, false for clones

    // Private clone constructor – creates a local copy with no backing region.
    shared_var(T const& t, bool /*clone_tag*/)
      : data_(t), owns_region_(false)
    {}

    // ---- helpers for region I/O ------------------------------------------

    void create_region()
    {
        Runtime* rt = legion_runtime();
        Context  ctx = legion_context();
        Rect<1> rect(0, 0);
        is_ = rt->create_index_space(ctx, rect);
        fs_ = rt->create_field_space(ctx);
        {
            FieldAllocator fa = rt->create_field_allocator(ctx, fs_);
            fa.allocate_field(sizeof(T), FID_VAL);
        }
        region_ = rt->create_logical_region(ctx, is_, fs_);
    }

    void init_region(T const& val)
    {
        Runtime* rt = legion_runtime();
        Context  ctx = legion_context();
        RegionRequirement req(region_, WRITE_DISCARD, EXCLUSIVE, region_);
        req.add_field(FID_VAL);
        InlineLauncher il(req);
        PhysicalRegion pr = rt->map_region(ctx, il);
        pr.wait_until_valid();
        const FieldAccessor<WRITE_DISCARD, T, 1, coord_t,
              Realm::AffineAccessor<T, 1, coord_t>> acc(pr, FID_VAL);
        acc[0] = val;
        rt->unmap_region(ctx, pr);
    }

    T read_region() const
    {
        Runtime* rt = legion_runtime();
        Context  ctx = legion_context();
        RegionRequirement req(region_, READ_ONLY, EXCLUSIVE, region_);
        req.add_field(FID_VAL);
        InlineLauncher il(req);
        PhysicalRegion pr = rt->map_region(ctx, il);
        pr.wait_until_valid();
        const FieldAccessor<READ_ONLY, T, 1, coord_t,
              Realm::AffineAccessor<T, 1, coord_t>> acc(pr, FID_VAL);
        T val = acc[0];
        rt->unmap_region(ctx, pr);
        return val;
    }

    void write_region(T const& val)
    {
        Runtime* rt = legion_runtime();
        Context  ctx = legion_context();
        RegionRequirement req(region_, READ_WRITE, EXCLUSIVE, region_);
        req.add_field(FID_VAL);
        InlineLauncher il(req);
        PhysicalRegion pr = rt->map_region(ctx, il);
        pr.wait_until_valid();
        const FieldAccessor<READ_WRITE, T, 1, coord_t,
              Realm::AffineAccessor<T, 1, coord_t>> acc(pr, FID_VAL);
        acc[0] = val;
        rt->unmap_region(ctx, pr);
    }

  public:
    // ---- constructors / destructor ---------------------------------------

    shared_var() : data_(), owns_region_(true)
    {
        create_region();
        init_region(T());
    }

    shared_var(T const& t) : data_(t), owns_region_(true)
    {
        create_region();
        init_region(t);
    }

    shared_var(T&& t) : data_(std::move(t)), owns_region_(true)
    {
        create_region();
        init_region(data_);
    }

    // Copy constructor produces a clone (no region).
    shared_var(shared_var const& rhs) : data_(rhs.data_), owns_region_(false)
    {}

    ~shared_var()
    {
        if (owns_region_ && legion_runtime() != nullptr) {
            Runtime* rt = legion_runtime();
            Context  ctx = legion_context();
            rt->destroy_logical_region(ctx, region_);
            rt->destroy_field_space(ctx, fs_);
            rt->destroy_index_space(ctx, is_);
        }
    }

    // ---- shared_var_base interface ---------------------------------------

    // Atomically reads from the backing region and returns a clone (no region).
    shared_var_base* clone() const override
    {
        if (owns_region_) {
            T val = read_region();
            return new shared_var(val, true);
        }
        return new shared_var(data_, true);
    }

    // Read the current value.  For region-backed variables this performs an
    // inline map / read / unmap cycle (atomic).  For clones it returns data_.
    T read() const
    {
        if (owns_region_) return read_region();
        return data_;
    }

    // Direct write (no transaction).  Writes both the local cache and the
    // backing region.
    void write(T const& rhs)
    {
        data_ = rhs;
        if (owns_region_) write_region(rhs);
    }

    // Virtual write from another shared_var_base (used during commit).
    void write(shared_var_base const& rhs) override
    {
        T val = dynamic_cast<shared_var const*>(&rhs)->data_;
        write(val);
    }

    // Inline-map the backing region and return a region_lock (RAII guard).
    region_lock lock() const override
    {
        assert(owns_region_);
        return region_lock(region_);
    }

    bool operator==(shared_var_base const& rhs) const override
    {
        T my_val = owns_region_ ? read_region() : data_;
        return my_val == dynamic_cast<shared_var const*>(&rhs)->data_;
    }

    // ---- Legion-specific helpers -----------------------------------------

    bool has_region() const override { return owns_region_; }

    LogicalRegion get_region() const override
    {
        assert(owns_region_);
        return region_;
    }

    bool equals_with_pr(PhysicalRegion const& pr,
                        shared_var_base const& other) const override
    {
        const FieldAccessor<READ_ONLY, T, 1, coord_t,
              Realm::AffineAccessor<T, 1, coord_t>> acc(pr, FID_VAL);
        T current = acc[0];
        return current == dynamic_cast<shared_var const*>(&other)->data_;
    }

    void write_with_pr(PhysicalRegion const& pr,
                       shared_var_base const& other) override
    {
        T val = dynamic_cast<shared_var const*>(&other)->data_;
        const FieldAccessor<READ_WRITE, T, 1, coord_t,
              Realm::AffineAccessor<T, 1, coord_t>> acc(pr, FID_VAL);
        acc[0] = val;
        data_  = val;   // keep the local cache consistent
    }

    // ---- convenience -----------------------------------------------------

    local_var get_local(transaction& trans)
    {
        return local_var(&trans, this);
    }
};

// ---------------------------------------------------------------------------
// transaction – optimistic STM transaction backed by Legion region mappings.
// ---------------------------------------------------------------------------
struct transaction
{
    std::list<
        std::pair<
            // The shared variable we're reading from
            shared_var_base*
            // The value we read from the variable
          , std::shared_ptr<shared_var_base>
        >
    > read_list;

    std::set<
        shared_var_base* // The shared variable we're writing to
    > write_set;

    std::list<
        std::pair<
            void*                                  // unused (future placeholder)
          , std::function<void(transaction*)>      // async callback
        >
    > async_list;

    std::map<
        shared_var_base* // The shared variable
      , std::shared_ptr<shared_var_base> // Current value of the variable
    > variables;

    void clear()
    {
        read_list.clear();
        write_set.clear();
        async_list.clear();
        variables.clear();
    }

    // Commit the transaction using Legion inline mappings for exclusive access.
    //
    // Algorithm:
    //   1.) Inline-map all involved regions (exclusive access).
    //   2.) Compare recorded reads against current region values (fail if stale).
    //   3.) Write committed values into the mapped regions.
    //   4.) Execute async callbacks (synchronously).
    //   5.) Unmap all regions (release exclusive access).
    bool commit_transaction()
    {
        // 1.) Obtain exclusive access by inline-mapping every backing region.
        //     The variable map is sorted by pointer value, giving a deterministic
        //     mapping order and avoiding deadlocks.
        std::map<shared_var_base*, PhysicalRegion> mapped;

        for ( std::pair<shared_var_base* const, std::shared_ptr<shared_var_base> >& var
            : variables)
        {
            assert(var.first != NULL);

            if (var.first->has_region())
            {
                LogicalRegion lr = var.first->get_region();
                RegionRequirement req(lr, READ_WRITE, EXCLUSIVE, lr);
                req.add_field(FID_VAL);
                InlineLauncher il(req);
                PhysicalRegion pr = legion_runtime()->map_region(legion_context(), il);
                pr.wait_until_valid();
                mapped[var.first] = pr;
            }
        }

        // 2.) Validate recorded reads against the current region contents.
        for ( std::pair<shared_var_base*, std::shared_ptr<shared_var_base> > const& var
            : read_list)
        {
            assert(var.first != NULL);

            auto mit = mapped.find(var.first);
            bool equal;
            if (mit != mapped.end())
                equal = var.first->equals_with_pr(mit->second, *var.second);
            else
                equal = ((*var.first) == (*var.second));

            if (!equal)
            {
                // Unmap all regions and abort.
                for (auto& m : mapped)
                    legion_runtime()->unmap_region(legion_context(), m.second);

                clear();
                return false;
            }
        }

        // 3.) Perform writes from the internal map into the regions.
        for (shared_var_base* var : write_set)
        {
            assert(var != NULL);

            auto it = variables.find(var);
            assert(it != variables.end());

            auto mit = mapped.find(var);
            if (mit != mapped.end())
                var->write_with_pr(mit->second, *it->second);
            else
                var->write(*it->second);
        }

        // 4.) Execute async callbacks synchronously.
        for (auto& op : async_list)
        {
            op.second(this);
        }

        // 5.) Release exclusive access by unmapping all regions.
        for (auto& m : mapped)
            legion_runtime()->unmap_region(legion_context(), m.second);

        return true;
    }

    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != NULL);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base> > entry(var, 0);

        auto result = variables.insert(entry);

        if (result.second == true)
        {
            // First access: clone (reads atomically from the backing region).
            (*result.first).second.reset((*var).clone());

            // Record the read.
            read_list.push_back(*result.first);

            return (*(*result.first).second);
        }
        else
            // Already in the internal state.
            return (*(*result.first).second);
    }

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

    // Register an async callback.  In the Legion model these are executed
    // synchronously at commit time.
    void then(void* /*unused*/, std::function<void(transaction*)> F)
    {
        std::pair<void*, std::function<void(transaction*)> > entry(nullptr, F);
        async_list.push_back(entry);
    }
};

// ---------------------------------------------------------------------------
// Out-of-line definitions for shared_var<T>::local_var
// ---------------------------------------------------------------------------

template <typename T>
shared_var<T>::local_var::operator T const& () const
{
    return dynamic_cast<shared_var const*>(&trans_->read(var_))->data_;
}

template <typename T>
T shared_var<T>::local_var::get() const
{
    return dynamic_cast<shared_var const*>(&trans_->read(var_))->data_;
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
    shared_var tmp(rhs, true);   // local clone – no backing region
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

// ---------------------------------------------------------------------------
// Out-of-line definition for transaction_future::then
// ---------------------------------------------------------------------------

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_);
    trans_->then(nullptr, f);
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
