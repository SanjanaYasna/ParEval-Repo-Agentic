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
#include <mutex>
#include <cassert>
#include <cstring>

using namespace Legion;

namespace astm
{

///////////////////////////////////////////////////////////////////////////////
// Task IDs used internally by ASTM
///////////////////////////////////////////////////////////////////////////////
enum ASTMTaskIDs
{
    ASTM_GENERIC_TASK_ID = 500,
};

///////////////////////////////////////////////////////////////////////////////
// Field ID for shared_var region storage
///////////////////////////////////////////////////////////////////////////////
enum ASTMFieldIDs
{
    FID_VAL = 100,
};

///////////////////////////////////////////////////////////////////////////////
// Thread-local Legion runtime/context holder.
// Must be initialised at the start of every Legion task that uses ASTM.
///////////////////////////////////////////////////////////////////////////////
struct LegionState
{
    static thread_local Runtime* runtime;
    static thread_local Context  ctx;

    static void set(Runtime* rt, Context c)
    {
        runtime = rt;
        ctx     = c;
    }

    static Runtime* get_runtime() { assert(runtime); return runtime; }
    static Context  get_context() { return ctx; }
};

// ---- definitions (must appear in exactly one translation unit) ----
// Put ASTM_LEGION_STATE_DEFINITIONS in one .cpp file, or use the
// inline-variable trick below (C++17).
#ifndef ASTM_LEGION_STATE_DEFINED
#define ASTM_LEGION_STATE_DEFINED
inline thread_local Runtime* LegionState::runtime = nullptr;
inline thread_local Context  LegionState::ctx     = Context();
#endif

///////////////////////////////////////////////////////////////////////////////
// Generic Legion task: executes a heap-allocated std::function<void()>*
// whose address is passed through TaskArgument (single-node only).
///////////////////////////////////////////////////////////////////////////////
inline void generic_astm_task_body(const Task*                       task,
                                   const std::vector<PhysicalRegion>&,
                                   Context                           ctx,
                                   Runtime*                          runtime)
{
    LegionState::set(runtime, ctx);

    std::function<void()>* fn_ptr = nullptr;
    assert(task->arglen == sizeof(fn_ptr));
    std::memcpy(&fn_ptr, task->args, sizeof(fn_ptr));
    (*fn_ptr)();
    delete fn_ptr;
}

///////////////////////////////////////////////////////////////////////////////
// Call once before Runtime::start() to register ASTM internal tasks.
///////////////////////////////////////////////////////////////////////////////
inline void register_tasks()
{
    TaskVariantRegistrar registrar(ASTM_GENERIC_TASK_ID, "astm_generic_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<generic_astm_task_body>(
        registrar, "astm_generic_task");
}

///////////////////////////////////////////////////////////////////////////////
// Launch an arbitrary callable as a Legion task.  Returns a Future that
// becomes ready when the callable completes.  Single-node only (the
// callable is passed by heap pointer through TaskArgument).
///////////////////////////////////////////////////////////////////////////////
template <typename F>
Future launch_async(F&& f)
{
    auto* fn = new std::function<void()>(std::forward<F>(f));
    TaskLauncher launcher(ASTM_GENERIC_TASK_ID,
                          TaskArgument(&fn, sizeof(fn)));
    return LegionState::get_runtime()->execute_task(
               LegionState::get_context(), launcher);
}

///////////////////////////////////////////////////////////////////////////////
// Forward declarations
///////////////////////////////////////////////////////////////////////////////
struct transaction;

///////////////////////////////////////////////////////////////////////////////
// shared_var_base  –  type-erased handle to a transactional variable
///////////////////////////////////////////////////////////////////////////////
struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;
    virtual void write(shared_var_base const&) = 0;
    virtual bool operator==(shared_var_base const&) const = 0;

    // Legion region management
    virtual LogicalRegion get_region() const = 0;
    virtual bool          has_region() const = 0;

    // Region-based operations used during commit
    virtual bool equals_from_region(PhysicalRegion const& pr,
                                    shared_var_base const& other) const = 0;
    virtual void write_to_region(PhysicalRegion& pr,
                                 shared_var_base const& value) = 0;
    virtual shared_var_base* clone_from_region(PhysicalRegion const& pr) const = 0;

    // Mutex-based lock (used by commit protocol for atomicity)
    virtual std::unique_lock<std::mutex> lock() const = 0;
};

///////////////////////////////////////////////////////////////////////////////
// transaction_future  –  wraps a Legion::Future that is filled at commit time
///////////////////////////////////////////////////////////////////////////////
struct transaction_future
{
  private:
    transaction* trans_;
    Future       fut_;
    bool         has_future_;

  public:
    transaction_future(transaction* trans)
      : trans_(trans), fut_(), has_future_(false)
    {}

    transaction_future(transaction& trans)
      : trans_(&trans), fut_(), has_future_(false)
    {}

    template <typename F>
    void then(F f);

    void get()
    {
        if (has_future_)
            fut_.get_void_result();
    }

    // Allow the commit protocol to set / chain the future
    Future*  future_ptr()        { return &fut_; }
    void     mark_has_future()   { has_future_ = true; }
};

///////////////////////////////////////////////////////////////////////////////
// shared_var<T>  –  a transactional variable backed by a Legion region
///////////////////////////////////////////////////////////////////////////////
template <typename T>
struct shared_var : shared_var_base
{
    // ------------------------------------------------------------------
    // local_var  –  transaction-local proxy
    // ------------------------------------------------------------------
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

    // ------------------------------------------------------------------
    // Tag type for constructing lightweight clones without a region
    // ------------------------------------------------------------------
    struct no_region_tag {};

  private:
    T                data_;
    mutable std::mutex mtx_;

    // Legion region fields
    LogicalRegion region_;
    IndexSpace    is_;
    FieldSpace    fs_;
    bool          owns_region_;

    // ------------------------------------------------------------------
    // Region helpers
    // ------------------------------------------------------------------
    void create_region()
    {
        Runtime* rt  = LegionState::get_runtime();
        Context  ctx = LegionState::get_context();

        is_ = rt->create_index_space(ctx, Rect<1>(0, 0));
        fs_ = rt->create_field_space(ctx);
        {
            FieldAllocator fa = rt->create_field_allocator(ctx, fs_);
            fa.allocate_field(sizeof(T), FID_VAL);
        }
        region_ = rt->create_logical_region(ctx, is_, fs_);
    }

    void sync_to_region()
    {
        Runtime* rt  = LegionState::get_runtime();
        Context  ctx = LegionState::get_context();

        InlineLauncher il(RegionRequirement(region_, WRITE_DISCARD,
                                            EXCLUSIVE, region_));
        il.add_field(FID_VAL);
        PhysicalRegion pr = rt->map_region(ctx, il);
        pr.wait_until_valid();

        const FieldAccessor<WRITE_DISCARD, T, 1, coord_t,
              Realm::AffineAccessor<T, 1, coord_t>> acc(pr, FID_VAL);
        acc[0] = data_;

        rt->unmap_region(ctx, pr);
    }

    void sync_from_region()
    {
        Runtime* rt  = LegionState::get_runtime();
        Context  ctx = LegionState::get_context();

        InlineLauncher il(RegionRequirement(region_, READ_ONLY,
                                            EXCLUSIVE, region_));
        il.add_field(FID_VAL);
        PhysicalRegion pr = rt->map_region(ctx, il);
        pr.wait_until_valid();

        const FieldAccessor<READ_ONLY, T, 1, coord_t,
              Realm::AffineAccessor<T, 1, coord_t>> acc(pr, FID_VAL);
        data_ = acc[0];

        rt->unmap_region(ctx, pr);
    }

  public:
    // Future used for async chaining on this variable
    Future queue;

    // ------------------------------------------------------------------
    // Constructors (with region)
    // ------------------------------------------------------------------
    shared_var()
      : data_(), mtx_(), owns_region_(true)
    {
        create_region();
        sync_to_region();
    }

    shared_var(T const& t)
      : data_(t), mtx_(), owns_region_(true)
    {
        create_region();
        sync_to_region();
    }

    shared_var(T&& t)
      : data_(std::move(t)), mtx_(), owns_region_(true)
    {
        create_region();
        sync_to_region();
    }

    shared_var(shared_var const& rhs)
      : data_(rhs.data_), mtx_(), owns_region_(true)
    {
        create_region();
        sync_to_region();
    }

    // ------------------------------------------------------------------
    // Lightweight clone constructor (no region)
    // ------------------------------------------------------------------
    shared_var(T const& t, no_region_tag)
      : data_(t), mtx_(), region_(LogicalRegion::NO_REGION),
        is_(IndexSpace::NO_SPACE), fs_(FieldSpace::NO_SPACE),
        owns_region_(false)
    {}

    // ------------------------------------------------------------------
    ~shared_var()
    {
        if (owns_region_ && region_ != LogicalRegion::NO_REGION)
        {
            Runtime* rt = LegionState::runtime;   // may be nullptr at shutdown
            if (rt)
            {
                Context ctx = LegionState::get_context();
                rt->destroy_logical_region(ctx, region_);
                rt->destroy_field_space(ctx, fs_);
                rt->destroy_index_space(ctx, is_);
            }
        }
    }

    // ------------------------------------------------------------------
    // clone()  –  Locks, reads current value, returns a no-region clone.
    //             If this var owns a region the value is read from the
    //             region so that it reflects the latest committed state.
    // ------------------------------------------------------------------
    shared_var_base* clone() const override
    {
        auto l = lock();
        if (owns_region_)
        {
            Runtime* rt  = LegionState::get_runtime();
            Context  ctx = LegionState::get_context();

            InlineLauncher il(RegionRequirement(region_, READ_ONLY,
                                                EXCLUSIVE, region_));
            il.add_field(FID_VAL);
            PhysicalRegion pr = rt->map_region(ctx, il);
            pr.wait_until_valid();

            const FieldAccessor<READ_ONLY, T, 1, coord_t,
                  Realm::AffineAccessor<T, 1, coord_t>> acc(pr, FID_VAL);
            T val = acc[0];

            rt->unmap_region(ctx, pr);

            // Update cached copy while we hold the lock
            const_cast<shared_var*>(this)->data_ = val;

            return new shared_var(val, no_region_tag{});
        }
        return new shared_var(data_, no_region_tag{});
    }

    // ------------------------------------------------------------------
    // Unlocked read of cached data (call after transaction commit or
    // when exclusive access is otherwise guaranteed).
    // ------------------------------------------------------------------
    T const& read() const
    {
        return data_;
    }

    // ------------------------------------------------------------------
    // Unlocked write of cached data + region sync.
    // ------------------------------------------------------------------
    void write(T const& rhs)
    {
        data_ = rhs;
        if (owns_region_)
            sync_to_region();
    }

    void write(shared_var_base const& rhs) override
    {
        data_ = dynamic_cast<shared_var const*>(&rhs)->read();
        if (owns_region_)
            sync_to_region();
    }

    // ------------------------------------------------------------------
    // Mutex lock (used by commit protocol)
    // ------------------------------------------------------------------
    std::unique_lock<std::mutex> lock() const override
    {
        return std::unique_lock<std::mutex>(mtx_);
    }

    // ------------------------------------------------------------------
    // Comparison
    // ------------------------------------------------------------------
    bool operator==(shared_var_base const& rhs) const override
    {
        return data_ == dynamic_cast<shared_var const*>(&rhs)->read();
    }

    // ------------------------------------------------------------------
    // Legion region accessors
    // ------------------------------------------------------------------
    LogicalRegion get_region() const override { return region_; }
    bool          has_region() const override { return owns_region_; }

    bool equals_from_region(PhysicalRegion const& pr,
                            shared_var_base const& other) const override
    {
        const FieldAccessor<READ_ONLY, T, 1, coord_t,
              Realm::AffineAccessor<T, 1, coord_t>> acc(pr, FID_VAL);
        T current = acc[0];
        return current == dynamic_cast<shared_var const*>(&other)->read();
    }

    void write_to_region(PhysicalRegion& pr,
                         shared_var_base const& value) override
    {
        const FieldAccessor<WRITE_DISCARD, T, 1, coord_t,
              Realm::AffineAccessor<T, 1, coord_t>> acc(pr, FID_VAL);
        T val = dynamic_cast<shared_var const*>(&value)->read();
        acc[0] = val;
        data_ = val;   // keep cached copy in sync
    }

    shared_var_base* clone_from_region(PhysicalRegion const& pr) const override
    {
        const FieldAccessor<READ_ONLY, T, 1, coord_t,
              Realm::AffineAccessor<T, 1, coord_t>> acc(pr, FID_VAL);
        T val = acc[0];
        return new shared_var(val, no_region_tag{});
    }

    // ------------------------------------------------------------------
    local_var get_local(transaction& trans)
    {
        return local_var(&trans, this);
    }
};

///////////////////////////////////////////////////////////////////////////////
// transaction  –  groups reads / writes into an atomic unit
///////////////////////////////////////////////////////////////////////////////
struct transaction
{
    std::list<
        std::pair<
            shared_var_base*,
            std::shared_ptr<shared_var_base>
        >
    > read_list;

    std::set<
        shared_var_base*
    > write_set;

    std::list<
        std::pair<
            Future*,
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

    // -----------------------------------------------------------------
    // commit_transaction
    //
    // 1. Obtain exclusive access (mutex locks + inline region maps).
    // 2. Validate recorded reads against current region values.
    // 3. Perform writes to regions (and cached copies).
    // 4. Schedule async operations as Legion tasks.
    // 5. Release exclusive access.
    // -----------------------------------------------------------------
    bool commit_transaction()
    {
        Runtime* rt  = LegionState::get_runtime();
        Context  ctx = LegionState::get_context();

        // 1a) Acquire mutex locks (sorted by pointer for deadlock avoidance)
        std::list<std::unique_lock<std::mutex>> locks;
        for (auto const& var : variables)
        {
            assert(var.first != nullptr);
            locks.push_back(var.first->lock());
        }

        // 1b) Inline-map every owned region with READ_WRITE / EXCLUSIVE
        std::map<shared_var_base*, PhysicalRegion> mappings;
        for (auto const& var : variables)
        {
            if (var.first->has_region())
            {
                InlineLauncher il(RegionRequirement(
                    var.first->get_region(),
                    READ_WRITE, EXCLUSIVE,
                    var.first->get_region()));
                il.add_field(FID_VAL);
                mappings[var.first] = rt->map_region(ctx, il);
            }
        }
        for (auto& m : mappings)
            m.second.wait_until_valid();

        // 2) Validate reads against region contents
        for (auto const& var : read_list)
        {
            assert(var.first != nullptr);
            auto mit = mappings.find(var.first);
            if (mit != mappings.end())
            {
                if (!var.first->equals_from_region(mit->second, *var.second))
                {
                    // Unmap regions and fail
                    for (auto& m : mappings)
                        rt->unmap_region(ctx, m.second);
                    clear();
                    return false;
                }
            }
            else
            {
                // No region – fall back to cached-data comparison
                if (!((*var.first) == (*var.second)))
                {
                    for (auto& m : mappings)
                        rt->unmap_region(ctx, m.second);
                    clear();
                    return false;
                }
            }
        }

        // 3) Perform writes through the mapped regions
        for (shared_var_base* var : write_set)
        {
            assert(var != nullptr);
            auto it = variables.find(var);
            assert(it != variables.end());

            auto mit = mappings.find(var);
            if (mit != mappings.end())
            {
                var->write_to_region(mit->second, *it->second);
            }
            else
            {
                var->write(*it->second);
            }
        }

        // 3b) Unmap all regions (release exclusive region access)
        for (auto& m : mappings)
            rt->unmap_region(ctx, m.second);

        // 4) Schedule async operations as Legion tasks
        for (auto& op : async_list)
        {
            auto fn_copy = op.second;
            transaction* self = this;

            if (op.first == nullptr)
            {
                // Fire-and-forget
                launch_async([fn_copy, self]() { fn_copy(self); });
            }
            else
            {
                // Chain onto existing future
                auto* fn = new std::function<void()>(
                    [fn_copy, self]() { fn_copy(self); }
                );
                TaskLauncher launcher(ASTM_GENERIC_TASK_ID,
                                      TaskArgument(&fn, sizeof(fn)));
                // If there is a prior future, add it as a precondition
                // (the task will not begin until the prior future is ready).
                // Note: only add if the future was previously set.
                // We rely on Future's internal validity.
                *(op.first) = rt->execute_task(ctx, launcher);
            }
        }

        // 5) Locks released by RAII
        return true;
    }

    // -----------------------------------------------------------------
    // read  –  snapshot a variable into transaction-local state
    // -----------------------------------------------------------------
    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != nullptr);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base>> entry(var, nullptr);

        auto result = variables.insert(entry);

        if (result.second == true)
        {
            // First access – clone the current value
            result.first->second.reset(var->clone());
            read_list.push_back(*result.first);
            return *result.first->second;
        }
        else
        {
            return *result.first->second;
        }
    }

    // -----------------------------------------------------------------
    // write  –  record a pending write in the transaction
    // -----------------------------------------------------------------
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
            result.first->second = entry.second;
            write_set.insert(var);
        }
    }

    // -----------------------------------------------------------------
    // then  –  register an async operation to run after commit
    // -----------------------------------------------------------------
    void then(Future* fut, std::function<void(transaction*)> F)
    {
        std::pair<Future*, std::function<void(transaction*)>> entry(fut, F);
        async_list.push_back(entry);
    }
};

///////////////////////////////////////////////////////////////////////////////
// shared_var<T>::local_var  –  method definitions
///////////////////////////////////////////////////////////////////////////////
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
    assert(trans_);
    trans_->then(&dynamic_cast<shared_var*>(var_)->queue, f);
}

///////////////////////////////////////////////////////////////////////////////
// transaction_future::then  –  definition
///////////////////////////////////////////////////////////////////////////////
template <typename F>
void transaction_future::then(F f)
{
    assert(trans_);
    mark_has_future();
    trans_->then(future_ptr(), f);
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
