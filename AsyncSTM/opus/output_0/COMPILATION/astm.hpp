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

using namespace Legion;

// ---------------------------------------------------------------------------
// Global Legion runtime / context – must be initialised at the start of every
// top-level or child task by calling  init_legion(runtime, ctx).
// ---------------------------------------------------------------------------
inline Runtime*& legion_runtime()
{
    static Runtime* rt = nullptr;
    return rt;
}

inline Context& legion_context()
{
    static Context ctx;
    return ctx;
}

inline void init_legion(Runtime* rt, Context ctx)
{
    legion_runtime() = rt;
    legion_context() = ctx;
}

// Field ID used as a token inside the lock-region (value is irrelevant).
enum { FID_LOCK_TOKEN = 200 };

// ---------------------------------------------------------------------------
// LegionLock  –  RAII wrapper around an inline mapping that acts as a mutex.
//
// Constructing a LegionLock inline-maps the given LogicalRegion with
// READ_WRITE / EXCLUSIVE coherence, which blocks until the runtime can
// guarantee exclusive access.  Destroying (or moved-from) the object
// unmaps the region, releasing the "lock".
// ---------------------------------------------------------------------------
struct LegionLock
{
    Runtime*       rt_;
    Context        ctx_;
    PhysicalRegion pr_;
    bool           active_;

    LegionLock()
      : rt_(nullptr), active_(false) {}

    LegionLock(Runtime* rt, Context ctx, LogicalRegion lr)
      : rt_(rt), ctx_(ctx), active_(true)
    {
        InlineLauncher il(
            RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
        il.add_field(FID_LOCK_TOKEN);
        pr_ = rt_->map_region(ctx_, il);
        pr_.wait_until_valid();
    }

    ~LegionLock()
    {
        if (active_ && rt_)
            rt_->unmap_region(ctx_, pr_);
    }

    // Move
    LegionLock(LegionLock&& o) noexcept
      : rt_(o.rt_), ctx_(o.ctx_), pr_(o.pr_), active_(o.active_)
    { o.active_ = false; }

    LegionLock& operator=(LegionLock&& o) noexcept
    {
        if (this != &o) {
            if (active_ && rt_) rt_->unmap_region(ctx_, pr_);
            rt_     = o.rt_;
            ctx_    = o.ctx_;
            pr_     = o.pr_;
            active_ = o.active_;
            o.active_ = false;
        }
        return *this;
    }

    // Non-copyable
    LegionLock(LegionLock const&)            = delete;
    LegionLock& operator=(LegionLock const&) = delete;
};

// ---------------------------------------------------------------------------
// shared_var_base  –  type-erased base for all shared transactional variables.
// ---------------------------------------------------------------------------
struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;

    virtual void write(shared_var_base const&) = 0;

    virtual LegionLock lock() const = 0;

    virtual bool operator==(shared_var_base const&) const = 0;
};

struct transaction;

// ---------------------------------------------------------------------------
// transaction_future  –  collects deferred actions that run after commit.
// ---------------------------------------------------------------------------
struct transaction_future
{
  private:
    transaction* trans_;
    std::vector<std::function<void(transaction*)>> deferred_;

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
        for (auto& fn : deferred_)
            fn(trans_);
        deferred_.clear();
    }

    // Allow the transaction to push deferred work here.
    std::vector<std::function<void(transaction*)>>& deferred_list()
    { return deferred_; }
};

// ---------------------------------------------------------------------------
// shared_var<T>
//
// Data is kept in an ordinary member (data_) for fast, direct access while
// the variable is "locked".  A one-element LogicalRegion serves as a
// synchronisation token: inline-mapping it with EXCLUSIVE coherence is the
// Legion equivalent of acquiring a mutex.
// ---------------------------------------------------------------------------
template <typename T>
struct shared_var : shared_var_base
{
    // Tag type to construct a clone (no Legion region).
    struct no_region_tag {};

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
    T               data_;
    LogicalRegion   region_;
    IndexSpaceT<1>  is_;
    FieldSpace      fs_;
    bool            owns_region_;

    void create_region()
    {
        Runtime* rt  = legion_runtime();
        Context  ctx = legion_context();
        assert(rt != nullptr);

        is_ = rt->create_index_space(ctx, Rect<1>(0, 0));
        fs_ = rt->create_field_space(ctx);
        {
            FieldAllocator fa = rt->create_field_allocator(ctx, fs_);
            fa.allocate_field(sizeof(int), FID_LOCK_TOKEN);
        }
        region_ = rt->create_logical_region(ctx, is_, fs_);
    }

  public:
    // ---- constructors -----------------------------------------------------
    shared_var()
      : data_(), owns_region_(true)
    { create_region(); }

    shared_var(T const& t)
      : data_(t), owns_region_(true)
    { create_region(); }

    shared_var(T&& t)
      : data_(std::move(t)), owns_region_(true)
    { create_region(); }

    shared_var(shared_var const& rhs)
      : data_(rhs.data_), owns_region_(true)
    { create_region(); }

    // Clone-only constructor – no Legion region created.
    shared_var(T const& t, no_region_tag)
      : data_(t), owns_region_(false)
    {}

    ~shared_var()
    {
        if (owns_region_ && legion_runtime() != nullptr)
        {
            Runtime* rt  = legion_runtime();
            Context  ctx = legion_context();
            rt->destroy_logical_region(ctx, region_);
            rt->destroy_field_space(ctx, fs_);
            rt->destroy_index_space(ctx, is_);
        }
    }

    // ---- shared_var_base interface ----------------------------------------

    // Locks, then copies data_.
    shared_var_base* clone() const override
    {
        if (owns_region_) {
            auto l = lock();
            return new shared_var(data_, no_region_tag{});
        }
        return new shared_var(data_, no_region_tag{});
    }

    // Doesn't lock (caller holds the lock via commit_transaction).
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
    void write(shared_var_base const& rhs) override
    {
        data_ = dynamic_cast<shared_var const*>(&rhs)->read();
    }

    LegionLock lock() const override
    {
        if (owns_region_)
            return LegionLock(legion_runtime(), legion_context(), region_);
        return LegionLock();   // no-op for clones
    }

    bool operator==(shared_var_base const& rhs) const override
    {
        return data_ == dynamic_cast<shared_var const*>(&rhs)->read();
    }

    // ---- transaction helpers ----------------------------------------------

    local_var get_local(transaction& trans)
    {
        return local_var(&trans, this);
    }
};

// ---------------------------------------------------------------------------
// transaction
//
// Mirrors the original STM transaction.  Locking is provided by LegionLock
// (inline-mapping with EXCLUSIVE coherence).  The optimistic
// read-validate-write protocol is preserved so that direct (out-of-band)
// writes can be detected and the transaction retried.
// ---------------------------------------------------------------------------
struct transaction
{
    std::list<
        std::pair<
            shared_var_base*                       // shared variable
          , std::shared_ptr<shared_var_base>       // snapshot at read time
        >
    > read_list;

    std::set<
        shared_var_base*                           // shared variable
    > write_set;

    // Async / deferred actions registered via then().
    // first  : pointer to a deferred-list (NULL → fire-and-forget)
    // second : the action
    std::list<
        std::pair<
            std::vector<std::function<void(transaction*)>>*
          , std::function<void(transaction*)>
        >
    > async_list;

    std::map<
        shared_var_base*
      , std::shared_ptr<shared_var_base>
    > variables;

    void clear()
    {
        read_list.clear();
        write_set.clear();
        async_list.clear();
        variables.clear();
    }

    // ------------------------------------------------------------------
    // commit_transaction
    //
    // 1.  Obtain exclusive access to every touched variable (inline-map
    //     each variable's lock-region with EXCLUSIVE coherence).
    // 2.  Validate recorded reads against current values.
    // 3.  Perform writes from the internal map.
    // 4.  Execute / defer async actions.
    // 5.  Release exclusive access (LegionLocks destroyed by RAII).
    // ------------------------------------------------------------------
    bool commit_transaction()
    {
        // 1. Lock all variables (sorted by pointer → consistent order).
        std::list<LegionLock> locks;

        for (auto const& var : variables)
        {
            assert(var.first != nullptr);
            locks.push_back((*var.first).lock());
        }

        // 2. Validate reads.
        for (auto const& var : read_list)
        {
            assert(var.first != nullptr);

            if (!((*var.first) == (*var.second)))
            {
                clear();
                return false;  // Conflict – caller should retry.
            }
        }

        // 3. Perform writes.
        for (shared_var_base* var : write_set)
        {
            assert(var != nullptr);
            auto it = variables.find(var);
            assert(it != variables.end());
            (*var).write((*(*it).second));
        }

        // 4. Async / deferred actions.
        for (auto& op : async_list)
        {
            if (op.first == nullptr)
            {
                // Fire-and-forget: execute synchronously.
                op.second(this);
            }
            else
            {
                // Deferred: push onto the provided list for later get().
                op.first->push_back(op.second);
            }
        }

        // 5. Locks released when `locks` goes out of scope.
        return true;
    }

    // ------------------------------------------------------------------
    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != nullptr);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base>>
            entry(var, nullptr);

        auto result = variables.insert(entry);

        if (result.second)
        {
            // First access – clone the current value (performs a lock inside).
            (*result.first).second.reset((*var).clone());

            // Record the read.
            read_list.push_back(*result.first);

            return (*(*result.first).second);
        }
        else
        {
            return (*(*result.first).second);
        }
    }

    // ------------------------------------------------------------------
    void write(shared_var_base* var, shared_var_base const& value)
    {
        assert(var != nullptr);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base>>
            entry(var, std::shared_ptr<shared_var_base>(value.clone()));

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

    // ------------------------------------------------------------------
    // Register a deferred action.
    // If deferred is NULL → fire-and-forget (runs during commit).
    // Otherwise the action is appended to *deferred during commit;
    // the caller retrieves and executes it later (see transaction_future).
    // ------------------------------------------------------------------
    void then(std::vector<std::function<void(transaction*)>>* deferred,
              std::function<void(transaction*)> F)
    {
        async_list.push_back({deferred, std::move(F)});
    }
};

// ---------------------------------------------------------------------------
// shared_var<T>::local_var  out-of-line definitions
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
    shared_var tmp(rhs, typename shared_var::no_region_tag{});
    trans_->write(var_, tmp);
    return *this;
}

template <typename T>
template <typename F>
void shared_var<T>::local_var::then(F f)
{
    // NOTE: mirrors original (rarely used path).
    assert(trans_);
    trans_->then(nullptr, f);
}

// ---------------------------------------------------------------------------
// transaction_future  out-of-line definition
// ---------------------------------------------------------------------------

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_);
    trans_->then(&deferred_, f);
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
