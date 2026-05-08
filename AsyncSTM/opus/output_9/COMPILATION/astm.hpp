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

#include <legion.h>

#include <list>
#include <set>
#include <map>
#include <vector>
#include <functional>
#include <memory>
#include <cassert>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <cstring>
#include <fstream>
#include <iostream>

#ifndef ASTM_TEST
#include <cassert>
#define ASTM_TEST assert
#endif

#ifndef ASTM_REPORT
#define ASTM_REPORT 0
#endif

using namespace Legion;

namespace astm
{

// ============================================================
// Thread-local Legion runtime / context management.
// Every Legion task entry point must call init_legion_context().
// ============================================================
namespace detail
{
    inline thread_local Runtime* tl_runtime = nullptr;
    inline thread_local Context  tl_context;

    // --------------------------------------------------------
    // Function registry: allows arbitrary std::function objects
    // to be executed inside a pre-registered Legion task.
    // --------------------------------------------------------
    inline std::mutex                                             func_mtx;
    inline std::unordered_map<uint64_t, std::function<void()>>   func_map;
    inline std::atomic<uint64_t>                                  func_id_ctr{0};

    inline uint64_t register_func(std::function<void()> fn)
    {
        std::lock_guard<std::mutex> lk(func_mtx);
        uint64_t id = func_id_ctr.fetch_add(1);
        func_map[id] = std::move(fn);
        return id;
    }

    inline std::function<void()> pop_func(uint64_t id)
    {
        std::lock_guard<std::mutex> lk(func_mtx);
        auto it = func_map.find(id);
        assert(it != func_map.end());
        auto fn = std::move(it->second);
        func_map.erase(it);
        return fn;
    }
} // namespace detail

inline void   init_legion_context(Runtime* rt, Context ctx)
{
    detail::tl_runtime = rt;
    detail::tl_context = ctx;
}
inline Runtime* get_runtime() { return detail::tl_runtime; }
inline Context  get_context() { return detail::tl_context; }

// ============================================================
// Task IDs used by ASTM
// ============================================================
enum AstmTaskIDs
{
    ASTM_ASYNC_WRAPPER_TASK_ID = 77700
};

// ============================================================
// The single pre-registered wrapper task.
// It optionally waits on predecessor futures (for chaining)
// then executes the function looked up in the registry.
// ============================================================
inline void astm_async_task_impl(const Task*                      task,
                                  const std::vector<PhysicalRegion>& /*regions*/,
                                  Context                           ctx,
                                  Runtime*                          runtime)
{
    init_legion_context(runtime, ctx);

    // Honor any predecessor futures that were attached for ordering.
    for (unsigned i = 0; i < task->futures.size(); ++i)
        task->futures[i].get_void_result();

    uint64_t func_id;
    assert(task->arglen == sizeof(func_id));
    std::memcpy(&func_id, task->args, sizeof(func_id));

    auto fn = detail::pop_func(func_id);
    fn();
}

/// Call once before Runtime::start(), e.g. from main() before starting Legion.
inline void register_astm_tasks()
{
    TaskVariantRegistrar registrar(ASTM_ASYNC_WRAPPER_TASK_ID,
                                   "astm_async_wrapper");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<astm_async_task_impl>(
        registrar, "astm_async_wrapper");
}

// ============================================================
// region_guard – RAII wrapper returned by shared_var::lock().
//
// On construction the logical region has already been inline-
// mapped READ_WRITE (exclusive).  On destruction the guard
// optionally executes a write-back functor (flushes the local
// data_ member back into the physical region) and then unmaps.
// ============================================================
struct region_guard
{
    PhysicalRegion                       pr;
    std::function<void(PhysicalRegion&)> writeback;
    bool                                 valid;

    region_guard()
      : valid(false)
    {}

    region_guard(PhysicalRegion p,
                 std::function<void(PhysicalRegion&)> wb = nullptr)
      : pr(p), writeback(std::move(wb)), valid(true)
    {}

    ~region_guard() { release(); }

    void release()
    {
        if (valid)
        {
            if (writeback) writeback(pr);
            if (detail::tl_runtime)
                detail::tl_runtime->unmap_region(detail::tl_context, pr);
            valid = false;
        }
    }

    region_guard(region_guard&& o) noexcept
      : pr(o.pr), writeback(std::move(o.writeback)), valid(o.valid)
    { o.valid = false; }

    region_guard& operator=(region_guard&& o) noexcept
    {
        if (this != &o)
        {
            release();
            pr        = o.pr;
            writeback = std::move(o.writeback);
            valid     = o.valid;
            o.valid   = false;
        }
        return *this;
    }

    region_guard(const region_guard&)            = delete;
    region_guard& operator=(const region_guard&) = delete;
};

// ============================================================
// legion_future – lightweight wrapper around Legion::Future.
//
// Provides .get() and .then() (implemented by launching a
// child task that depends on the predecessor future).
// ============================================================
struct legion_future
{
    Legion::Future fut;
    bool           has_value;

    legion_future()
      : has_value(false)
    {}

    explicit legion_future(Legion::Future f)
      : fut(f), has_value(true)
    {}

    void get()
    {
        if (has_value)
            fut.get_void_result();
    }

    /// Chain: returns a new future whose task runs after this
    /// future completes and executes fn.
    legion_future then(std::function<void()> fn)
    {
        uint64_t fid = detail::register_func(std::move(fn));
        TaskLauncher launcher(ASTM_ASYNC_WRAPPER_TASK_ID,
                              TaskArgument(&fid, sizeof(fid)));
        if (has_value)
            launcher.add_future(fut);
        Legion::Future result =
            get_runtime()->execute_task(get_context(), launcher);
        return legion_future(result);
    }
};

inline legion_future make_ready_future()
{
    return legion_future();          // default-constructed = already complete
}

/// Launch fn(...) as a Legion child task; return a future.
template <typename F, typename... Args>
legion_future legion_async(F&& f, Args&&... args)
{
    // Bind the callable + arguments into a void() functor and
    // register it so the wrapper task can look it up.
    auto bound = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    uint64_t fid = detail::register_func(
        [bound = std::move(bound)]() mutable { bound(); });
    TaskLauncher launcher(ASTM_ASYNC_WRAPPER_TASK_ID,
                          TaskArgument(&fid, sizeof(fid)));
    Legion::Future result =
        get_runtime()->execute_task(get_context(), launcher);
    return legion_future(result);
}

// ============================================================
// shared_var_base – virtual interface for the transaction.
// ============================================================
struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const            = 0;
    virtual void write(shared_var_base const&)        = 0;
    virtual region_guard lock() const                 = 0;
    virtual bool operator==(shared_var_base const&) const = 0;
};

struct transaction;

// ============================================================
// transaction_future
// ============================================================
struct transaction_future
{
    typedef legion_future future_type;

  private:
    transaction* trans_;
    future_type  fut_;

  public:
    transaction_future(transaction* trans)
      : trans_(trans)
      , fut_(make_ready_future())
    {}

    transaction_future(transaction& trans)
      : trans_(&trans)
      , fut_(make_ready_future())
    {}

    template <typename F>
    void then(F f);

    void get()
    {
        fut_.get();
    }
};

// ============================================================
// shared_var<T>
//
// Data is stored both in a local member (data_) and in a
// single-element LogicalRegion.  The region is the
// authoritative copy; data_ is kept synchronised at well-
// defined points (construction, clone, lock, direct write).
//
// Transaction-local copies ("clones") are light-weight
// objects that do NOT own a LogicalRegion.
// ============================================================
template <typename T>
struct shared_var : shared_var_base
{
    typedef legion_future future_type;

    // --------------------------------------------------------
    // local_var – proxy for transactional reads / writes
    // --------------------------------------------------------
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
    // Tag type for the "clone" constructor (no region created).
    struct clone_tag {};

    T                data_;
    LogicalRegion    lr_;
    IndexSpace       is_;
    FieldSpace       fs_;
    FieldID          fid_;
    bool             owns_region_;
    mutable bool     is_locked_;     // true while a lock() guard is alive

    // -- helpers for region I/O --------------------------------

    void create_and_init_region()
    {
        Runtime* rt = get_runtime();
        Context  ctx = get_context();

        if (!rt)
        {
            // Runtime not available – fall back to in-memory only.
            owns_region_ = false;
            lr_ = LogicalRegion::NO_REGION;
            return;
        }

        Rect<1> bounds(0, 0);   // single element
        is_ = rt->create_index_space(ctx, bounds);
        fs_ = rt->create_field_space(ctx);
        {
            FieldAllocator fa = rt->create_field_allocator(ctx, fs_);
            fid_ = fa.allocate_field(sizeof(T));
        }
        lr_ = rt->create_logical_region(ctx, is_, fs_);

        // Write initial value into the region.
        write_to_region(data_);
    }

    void write_to_region(T const& val) const
    {
        Runtime* rt = get_runtime();
        Context  ctx = get_context();
        if (!rt || lr_ == LogicalRegion::NO_REGION) return;

        InlineLauncher il(
            RegionRequirement(lr_, WRITE_DISCARD, EXCLUSIVE, lr_));
        il.add_field(fid_);
        PhysicalRegion pr = rt->map_region(ctx, il);
        pr.wait_until_valid();
        const FieldAccessor<WRITE_DISCARD, T, 1> acc(pr, fid_);
        acc[0] = val;
        rt->unmap_region(ctx, pr);
    }

    T read_from_region() const
    {
        Runtime* rt = get_runtime();
        Context  ctx = get_context();
        assert(rt && lr_ != LogicalRegion::NO_REGION);

        InlineLauncher il(
            RegionRequirement(lr_, READ_ONLY, EXCLUSIVE, lr_));
        il.add_field(fid_);
        PhysicalRegion pr = rt->map_region(ctx, il);
        pr.wait_until_valid();
        const FieldAccessor<READ_ONLY, T, 1> acc(pr, fid_);
        T val = acc[0];
        rt->unmap_region(ctx, pr);
        return val;
    }

  public:
    future_type queue;

    // -- constructors (create a LogicalRegion) -----------------

    shared_var()
      : data_(), owns_region_(true), is_locked_(false),
        lr_(LogicalRegion::NO_REGION),
        fid_(0), queue(make_ready_future())
    { create_and_init_region(); }

    shared_var(T const& t)
      : data_(t), owns_region_(true), is_locked_(false),
        lr_(LogicalRegion::NO_REGION),
        fid_(0), queue(make_ready_future())
    { create_and_init_region(); }

    shared_var(T&& t)
      : data_(std::move(t)), owns_region_(true), is_locked_(false),
        lr_(LogicalRegion::NO_REGION),
        fid_(0), queue(make_ready_future())
    { create_and_init_region(); }

    // -- clone constructor (no region) -------------------------

    shared_var(T const& t, clone_tag)
      : data_(t), lr_(LogicalRegion::NO_REGION),
        owns_region_(false), is_locked_(false),
        fid_(0), queue(make_ready_future())
    {}

    // Copy constructor – produces a light-weight clone (no region).
    shared_var(shared_var const& rhs)
      : data_(rhs.data_), lr_(LogicalRegion::NO_REGION),
        owns_region_(false), is_locked_(false),
        fid_(0), queue(make_ready_future())
    {}

    ~shared_var()
    {
        if (owns_region_ && lr_ != LogicalRegion::NO_REGION
            && detail::tl_runtime)
        {
            detail::tl_runtime->destroy_logical_region(
                detail::tl_context, lr_);
            detail::tl_runtime->destroy_field_space(
                detail::tl_context, fs_);
            detail::tl_runtime->destroy_index_space(
                detail::tl_context, is_);
        }
    }

    // -- shared_var_base interface ----------------------------

    /// Atomically reads the region and returns a clone (no region).
    shared_var_base* clone() const override
    {
        if (owns_region_ && lr_ != LogicalRegion::NO_REGION)
        {
            T val = read_from_region();
            return new shared_var(val, clone_tag{});
        }
        return new shared_var(data_, clone_tag{});
    }

    /// Read the local (in-memory) copy.  Guaranteed to be in
    /// sync after construction, clone(), lock(), or direct write().
    T const& read() const
    {
        return data_;
    }

    /// Modify only the local data_ member (used during commit,
    /// when the region is already mapped via a lock guard whose
    /// writeback will flush data_ to the region on release).
    void write(T const& rhs)
    {
        data_ = rhs;
        // If nobody holds the region locked, push to region now.
        if (owns_region_ && lr_ != LogicalRegion::NO_REGION
            && !is_locked_)
        {
            write_to_region(rhs);
        }
    }

    void write(shared_var_base const& rhs) override
    {
        data_ = dynamic_cast<shared_var const*>(&rhs)->read();
        if (owns_region_ && lr_ != LogicalRegion::NO_REGION
            && !is_locked_)
        {
            write_to_region(data_);
        }
    }

    /// Map the region READ_WRITE / EXCLUSIVE.  Returns an RAII
    /// guard that:
    ///   – on construction: reads region → data_
    ///   – on destruction : writes data_ → region, then unmaps
    ///
    /// While the guard exists no other task can access this
    /// variable's region (Legion serialises inline mappings
    /// with conflicting privileges).
    region_guard lock() const override
    {
        if (!owns_region_ || lr_ == LogicalRegion::NO_REGION)
            return region_guard();

        Runtime* rt  = get_runtime();
        Context  ctx = get_context();

        InlineLauncher il(
            RegionRequirement(lr_, READ_WRITE, EXCLUSIVE, lr_));
        il.add_field(fid_);
        PhysicalRegion pr = rt->map_region(ctx, il);
        pr.wait_until_valid();

        // Sync: region → data_
        {
            const FieldAccessor<READ_ONLY, T, 1> acc(pr, fid_);
            const_cast<T&>(data_) = acc[0];
        }

        is_locked_ = true;

        // Capture a raw pointer to data_ and the field id for the
        // write-back lambda.
        T*      dptr        = const_cast<T*>(&data_);
        FieldID f           = fid_;
        bool*   locked_flag = const_cast<bool*>(&is_locked_);

        auto wb = [dptr, f, locked_flag](PhysicalRegion& p)
        {
            const FieldAccessor<READ_WRITE, T, 1> acc(p, f);
            acc[0] = *dptr;
            *locked_flag = false;
        };

        return region_guard(pr, std::move(wb));
    }

    bool operator==(shared_var_base const& rhs) const override
    {
        return data_ ==
               dynamic_cast<shared_var const*>(&rhs)->read();
    }

    local_var get_local(transaction& trans)
    {
        return local_var(&trans, this);
    }
};

// ============================================================
// transaction
// ============================================================
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
            legion_future*,
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
        // 1.) Map every variable's region READ_WRITE / EXCLUSIVE
        //     (provides mutual exclusion – equivalent to locking).
        std::list<region_guard> locks;

        for ( auto const& var : variables)
        {
            assert(var.first != NULL);
            locks.push_back((*var.first).lock());
        }

        // 2.) Compare recorded reads against current region values.
        for ( auto const& var : read_list)
        {
            assert(var.first != NULL);

            if (!((*var.first) == (*var.second)))
            {
                clear();
                // Guards are destroyed → write-back + unmap (harmless
                // because data_ was not modified for written vars yet).
                return false;
            }
        }

        // 3.) Perform writes (modifies data_ of the real shared_var;
        //     the guard's write-back will flush to region on release).
        for (shared_var_base* var : write_set)
        {
            assert(var != NULL);

            auto it = variables.find(var);
            assert(it != variables.end());

            (*var).write((*(*it).second));
        }

        // 4.) Launch async operations as Legion child tasks.
        for ( auto& op : async_list)
        {
            if (op.first == NULL)
            {
                // Fire-and-forget.
                legion_async(op.second, this);
            }
            else
            {
                transaction* self = this;
                auto fn = op.second;
                (*op.first) = (*op.first).then(
                    [fn, self]() { fn(self); }
                );
            }
        }

        // 5.) Guards are destroyed → write-back + unmap (release
        //     exclusive access).
        return true;
    }

    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != NULL);

        std::pair<shared_var_base*,
                  std::shared_ptr<shared_var_base> > entry(var, nullptr);

        auto result = variables.insert(entry);

        if (result.second == true)
        {
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

        std::pair<shared_var_base*,
                  std::shared_ptr<shared_var_base> >
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

    void then(legion_future* fut,
              std::function<void(transaction*)> F)
    {
        async_list.push_back({fut, std::move(F)});
    }
};

// ============================================================
// shared_var<T>::local_var implementation
// ============================================================

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
    trans_->then(
        &dynamic_cast<shared_var*>(var_)->queue,
        std::function<void(transaction*)>(f));
}

template <typename F>
void transaction_future::then(F f)
{
    assert(trans_);
    trans_->then(
        &fut_,
        std::function<void(transaction*)>(f));
}

} // namespace astm

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
