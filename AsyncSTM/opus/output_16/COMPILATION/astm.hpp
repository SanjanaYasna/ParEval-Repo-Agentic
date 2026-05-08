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
#include <mutex>
#include <memory>
#include <atomic>
#include <cassert>
#include <cstring>
#include <fstream>

using namespace Legion;

// ============================================================================
// Legion execution model infrastructure for ASTM
// ============================================================================

namespace astm {

// ----------------------------------------------------------------------------
// Thread-local Legion context management.
// Every Legion task must call set_legion_context() at its entry point.
// ----------------------------------------------------------------------------
namespace detail {
    inline Runtime*& runtime_ref() {
        static thread_local Runtime* rt = nullptr;
        return rt;
    }
    inline Context& context_ref() {
        static thread_local Context ctx;
        return ctx;
    }
    inline Runtime* get_runtime() { return runtime_ref(); }
    inline Context  get_context() { return context_ref(); }
} // namespace detail

inline void set_legion_context(Runtime* rt, Context ctx) {
    detail::runtime_ref() = rt;
    detail::context_ref() = ctx;
}

// ----------------------------------------------------------------------------
// Async function registry.
// Legion tasks receive serialised TaskArgument, so we store the actual
// std::function<void()> in a global table keyed by a uint64_t id, then
// pass only the id through TaskArgument.
// ----------------------------------------------------------------------------
namespace detail {
    struct AsyncRegistry {
        std::mutex                                     mtx;
        std::map<uint64_t, std::function<void()>>      funcs;
        std::atomic<uint64_t>                          counter{0};

        static AsyncRegistry& instance() {
            static AsyncRegistry reg;
            return reg;
        }

        uint64_t add(std::function<void()> fn) {
            uint64_t id = counter.fetch_add(1);
            std::lock_guard<std::mutex> lk(mtx);
            funcs[id] = std::move(fn);
            return id;
        }

        std::function<void()> pop(uint64_t id) {
            std::lock_guard<std::mutex> lk(mtx);
            auto it = funcs.find(id);
            assert(it != funcs.end());
            auto fn = std::move(it->second);
            funcs.erase(it);
            return fn;
        }
    };
} // namespace detail

// ----------------------------------------------------------------------------
// Task IDs used by ASTM internally.
// ----------------------------------------------------------------------------
enum AstmTaskID : TaskID {
    ASTM_ASYNC_TASK_ID = 76543  // pick a high number to avoid collisions
};

// The single generic task body that the runtime invokes for every
// ASTM_ASYNC / .then() call.
static void astm_async_task_body(const Task*                     task,
                                 const std::vector<PhysicalRegion>&,
                                 Context                         ctx,
                                 Runtime*                        runtime)
{
    set_legion_context(runtime, ctx);
    uint64_t fn_id;
    assert(task->arglen == sizeof(uint64_t));
    std::memcpy(&fn_id, task->args, sizeof(uint64_t));
    auto fn = detail::AsyncRegistry::instance().pop(fn_id);
    fn();
}

/// Call once from main(), **before** Runtime::start().
inline void register_astm_tasks() {
    TaskVariantRegistrar registrar(ASTM_ASYNC_TASK_ID, "astm_async");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<astm_async_task_body>(
        registrar, "astm_async");
}

// ----------------------------------------------------------------------------
// Future wrapper  –  provides the .get() / .then() interface expected by the
// ASTM transaction code while delegating to Legion::Future<void>.
// ----------------------------------------------------------------------------
template <typename T>
class astm_future;

template <>
class astm_future<void> {
    Future fut_;
    bool   valid_;

public:
    astm_future() : fut_(), valid_(false) {}
    explicit astm_future(Future f) : fut_(f), valid_(true) {}

    astm_future(const astm_future&)            = default;
    astm_future& operator=(const astm_future&) = default;
    astm_future(astm_future&&)                 = default;
    astm_future& operator=(astm_future&&)      = default;

    void get() {
        if (valid_) {
            fut_.get_void_result();
            valid_ = false;
        }
    }

    /// Chain a continuation.  The callable \a f is invoked (with no
    /// arguments) after the predecessor future completes.  Returns a new
    /// future representing the completion of \a f.
    template <typename F>
    astm_future<void> then(F f) {
        // Capture predecessor state so the continuation can wait on it.
        bool   prev_valid = valid_;
        Future prev_fut   = fut_;

        std::function<void()> fn =
            [f, prev_valid, prev_fut]() mutable {
                if (prev_valid)
                    prev_fut.get_void_result();   // block until predecessor done
                f();                              // run continuation
            };

        uint64_t fn_id = detail::AsyncRegistry::instance().add(std::move(fn));

        TaskLauncher launcher(ASTM_ASYNC_TASK_ID,
                              TaskArgument(&fn_id, sizeof(fn_id)));
        if (prev_valid)
            launcher.add_future(prev_fut);        // hint to the runtime

        Runtime* rt  = detail::get_runtime();
        Context  ctx = detail::get_context();
        return astm_future<void>(rt->execute_task(ctx, launcher));
    }
};

// ----------------------------------------------------------------------------
// astm_async_launch  –  replacement for hpx::async / std::async.
// Launches a new Legion task that will execute f(args...).
// ----------------------------------------------------------------------------
template <typename F, typename... Args>
astm_future<void> astm_async_launch(F&& f, Args&&... args)
{
    auto bound = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    std::function<void()> fn(std::move(bound));
    uint64_t fn_id = detail::AsyncRegistry::instance().add(std::move(fn));

    TaskLauncher launcher(ASTM_ASYNC_TASK_ID,
                          TaskArgument(&fn_id, sizeof(fn_id)));

    Runtime* rt  = detail::get_runtime();
    Context  ctx = detail::get_context();
    return astm_future<void>(rt->execute_task(ctx, launcher));
}

/// Returns an already-satisfied future (no pending work).
inline astm_future<void> make_ready_future() {
    return astm_future<void>();
}

} // namespace astm

// ============================================================================
// Compatibility macros  –  map the ASTM_* tokens used throughout the code base
// to the Legion-backed implementations above.
// ============================================================================
#define ASTM_MUTEX              std::mutex
#define ASTM_LOCK               std::unique_lock<std::mutex>
#define ASTM_FUTURE             astm::astm_future
#define ASTM_FUNCTION           std::function
#define ASTM_ASYNC              astm::astm_async_launch
#define ASTM_MAKE_READY_FUTURE  astm::make_ready_future()

#define ASTM_TEST               assert
#define ASTM_REPORT             0

// ============================================================================
// ASTM core  –  shared variables, transactions, local variables
// ============================================================================

namespace astm
{

// Tag type used internally to create lightweight (no-region) copies for the
// transaction's snapshot / read-list entries.
struct no_region_tag_t {};
static constexpr no_region_tag_t no_region_tag{};

struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;

    virtual void write(shared_var_base const&) = 0;

    virtual ASTM_LOCK lock() const = 0;

    virtual bool operator==(shared_var_base const&) const = 0;
};

struct transaction;

struct transaction_future
{
    typedef ASTM_FUTURE<void> future_type;

  private:
    transaction* trans_;
    future_type  fut_;

  public:
    transaction_future(transaction* trans)
      : trans_(trans)
      , fut_(ASTM_MAKE_READY_FUTURE)
    {}

    transaction_future(transaction& trans)
      : trans_(&trans)
      , fut_(ASTM_MAKE_READY_FUTURE)
    {}

    template <typename F>
    void then(F f);

    void get()
    {
        fut_.get();
    }
};

// ============================================================================
// shared_var<T>
//
// Data lives in a Legion LogicalRegion (one element, one field of type T) **and**
// is cached locally in data_ for fast access during the STM commit protocol.
// The region is only created for "primary" shared variables; lightweight
// snapshot copies (produced by clone()) skip region creation.
// ============================================================================

template <typename T>
struct shared_var : shared_var_base
{
    typedef ASTM_FUTURE<void> future_type;

    struct local_var
    {
      private:
        transaction*     trans_;
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
    mutable ASTM_MUTEX mtx_;

    // Legion region backing (may be absent for snapshot copies).
    bool            has_region_;
    LogicalRegion   region_;
    IndexSpace      is_;
    FieldSpace      fs_;
    FieldID         fid_;

    // Helper: create the backing logical region and write the initial value.
    void init_region(T const& val)
    {
        Runtime* rt = detail::get_runtime();
        if (!rt) { has_region_ = false; return; }
        Context ctx = detail::get_context();

        Rect<1> bounds(Point<1>(0), Point<1>(0));
        is_ = rt->create_index_space(ctx, bounds);

        fs_ = rt->create_field_space(ctx);
        {
            FieldAllocator fa = rt->create_field_allocator(ctx, fs_);
            fid_ = fa.allocate_field(sizeof(T));
        }

        region_ = rt->create_logical_region(ctx, is_, fs_);
        has_region_ = true;

        // Inline-map to write the initial value.
        {
            InlineLauncher il(
                RegionRequirement(region_, WRITE_DISCARD, EXCLUSIVE, region_));
            il.add_field(fid_);
            PhysicalRegion pr = rt->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<WRITE_DISCARD, T, 1> acc(pr, fid_);
            acc[Point<1>(0)] = val;
            rt->unmap_region(ctx, pr);
        }
    }

    // Helper: push the cached value back to the region.
    void sync_to_region()
    {
        if (!has_region_) return;
        Runtime* rt = detail::get_runtime();
        if (!rt) return;
        Context ctx = detail::get_context();

        InlineLauncher il(
            RegionRequirement(region_, WRITE_DISCARD, EXCLUSIVE, region_));
        il.add_field(fid_);
        PhysicalRegion pr = rt->map_region(ctx, il);
        pr.wait_until_valid();
        const FieldAccessor<WRITE_DISCARD, T, 1> acc(pr, fid_);
        acc[Point<1>(0)] = data_;
        rt->unmap_region(ctx, pr);
    }

  public:
    future_type queue;

    // --- Constructors -------------------------------------------------------

    shared_var()
      : data_(), mtx_(), has_region_(false), fid_(0),
        queue(ASTM_MAKE_READY_FUTURE)
    {
        init_region(data_);
    }

    shared_var(T const& t)
      : data_(t), mtx_(), has_region_(false), fid_(0),
        queue(ASTM_MAKE_READY_FUTURE)
    {
        init_region(data_);
    }

    shared_var(T&& t)
      : data_(t), mtx_(), has_region_(false), fid_(0),
        queue(ASTM_MAKE_READY_FUTURE)
    {
        init_region(data_);
    }

    // Copy constructor – creates a new primary shared_var with its own region.
    shared_var(shared_var const& rhs)
      : data_(rhs.data_), mtx_(), has_region_(false), fid_(0),
        queue(ASTM_MAKE_READY_FUTURE)
    {
        init_region(data_);
    }

    // Lightweight snapshot constructor (no region) – used by clone().
    shared_var(T const& t, no_region_tag_t)
      : data_(t), mtx_(), has_region_(false), fid_(0),
        queue(ASTM_MAKE_READY_FUTURE)
    {}

    ~shared_var()
    {
        if (has_region_) {
            Runtime* rt = detail::get_runtime();
            if (rt) {
                Context ctx = detail::get_context();
                rt->destroy_logical_region(ctx, region_);
                rt->destroy_field_space(ctx, fs_);
                rt->destroy_index_space(ctx, is_);
            }
        }
    }

    // --- shared_var_base interface ------------------------------------------

    // Locks, reads current value, returns a lightweight (no-region) copy.
    shared_var_base* clone() const
    {
        auto l = lock();
        return new shared_var(data_, no_region_tag);
    }

    // Doesn't lock.
    T const& read() const
    {
        return data_;
    }

    // Doesn't lock – direct write to cache (bypasses the region).
    void write(T const& rhs)
    {
        data_ = rhs;
        sync_to_region();
    }

    // Doesn't lock – virtual override.
    void write(shared_var_base const& rhs)
    {
        data_ = dynamic_cast<shared_var const*>(&rhs)->read();
        sync_to_region();
    }

    ASTM_LOCK lock() const
    {
        return ASTM_LOCK(mtx_);
    }

    bool operator==(shared_var_base const& rhs) const
    {
        return data_ == dynamic_cast<shared_var const*>(&rhs)->read();
    }

    local_var get_local(transaction& trans)
    {
        return local_var(&trans, this);
    }

    // ---- Legion-specific accessors -----------------------------------------

    bool            has_region() const { return has_region_; }
    LogicalRegion   get_region() const { return region_; }
    FieldID         get_field()  const { return fid_; }
};

// ============================================================================
// transaction
// ============================================================================

struct transaction
{
    std::list<
        std::pair<
            shared_var_base*                       // shared variable
          , std::shared_ptr<shared_var_base>        // value we read
        >
    > read_list;

    std::set<
        shared_var_base*
    > write_set;

    std::list<
        std::pair<
            ASTM_FUTURE<void>*
          , ASTM_FUNCTION<void(transaction*)>
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

    bool commit_transaction()
    {
        // 1.) Obtain exclusive access to all the variables.
        std::list<ASTM_LOCK> locks;

        for ( std::pair<shared_var_base*, std::shared_ptr<shared_var_base> > const& var
            : variables)
        {
            assert(var.first != NULL);
            locks.push_back((*var.first).lock());
        }

        // 2.) Compare our recorded reads against the current values.
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

        // 3.) Perform writes, reading from our internal map.
        for (shared_var_base* var : write_set)
        {
            assert(var != NULL);

            auto it = variables.find(var);
            assert(it != variables.end());

            (*var).write((*(*it).second));
        }

        // 4.) Perform async operations.
        for ( std::pair<ASTM_FUTURE<void>*, ASTM_FUNCTION<void(transaction*)> >& op
            : async_list)
        {
            if (op.first == NULL)
                ASTM_ASYNC(op.second, this);
            else
                (*op.first) = (*op.first).then(std::bind(op.second, this));
        }

        // 5.) Release exclusive access (RAII via locks list destruction).
        return true;
    }

    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != NULL);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base> > entry(var, 0);

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

    void then(ASTM_FUTURE<void>* fut, ASTM_FUNCTION<void(transaction*)> F)
    {
        std::pair<ASTM_FUTURE<void>*, ASTM_FUNCTION<void(transaction*)> > entry(fut, F);
        async_list.push_back(entry);
    }
};

// ============================================================================
// local_var method definitions (need full transaction definition)
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
    shared_var tmp(rhs, no_region_tag);   // lightweight snapshot – no region
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

#endif // ASTM_DBA88345_57B8_4CC8_A574_D5F007250E94
