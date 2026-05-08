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
#include <mutex>
#include <future>
#include <cassert>
#include <fstream>

namespace astm
{

using namespace Legion;

// =========================================================================
// Thread-local Legion context – every Legion task must call
// set_legion_context(runtime, ctx) at entry.
// =========================================================================
inline thread_local Runtime* legion_runtime_tl = nullptr;
inline thread_local Context  legion_ctx_tl;

inline void set_legion_context(Runtime* rt, Context ctx)
{
    legion_runtime_tl = rt;
    legion_ctx_tl     = ctx;
}
inline Runtime* get_legion_runtime() { return legion_runtime_tl; }
inline Context  get_legion_context()  { return legion_ctx_tl; }

// =========================================================================
// Field ID shared by every shared_var region
// =========================================================================
enum { ASTM_FID_VALUE = 200 };

// =========================================================================
// legion_future – a lightweight future wrapper that supports .then(), .get()
// and default construction (ready state).  Only void specialisation is needed.
// =========================================================================
template <typename T> struct legion_future;

template <>
struct legion_future<void>
{
private:
    std::shared_future<void> impl_;

public:
    // Default-constructed future is immediately ready.
    legion_future()
    {
        std::promise<void> p;
        p.set_value();
        impl_ = p.get_future().share();
    }

    legion_future(std::future<void>&& f)
      : impl_(f.share())
    {}

    legion_future(std::shared_future<void> f)
      : impl_(std::move(f))
    {}

    void get()
    {
        if (impl_.valid())
            impl_.get();
    }

    bool valid() const { return impl_.valid(); }

    // Wait for the previous stage, then execute f, return a new future.
    template <typename F>
    legion_future<void> then(F f)
    {
        auto prev = impl_;
        auto new_fut = std::async(std::launch::async,
            [prev, f]() mutable {
                if (prev.valid()) prev.wait();
                f();
            });
        return legion_future<void>(std::move(new_fut));
    }
};

// =========================================================================
// make_ready_future – returns an already-satisfied legion_future<void>
// =========================================================================
inline legion_future<void> make_ready_future()
{
    return legion_future<void>();          // default ctor is ready
}

// =========================================================================
// Async worker infrastructure – lets us launch arbitrary std::function<void()>
// callables as Legion child tasks.
// =========================================================================
enum { ASTM_WORKER_TASK_ID = 77700 };

// Shared work queue (single-node, protected by mutex)
struct work_item
{
    std::function<void()> func;
    std::promise<void>    prom;
};

inline std::mutex& work_queue_mutex()
{
    static std::mutex m;
    return m;
}
inline std::map<uint64_t, std::shared_ptr<work_item>>& work_queue_map()
{
    static std::map<uint64_t, std::shared_ptr<work_item>> m;
    return m;
}
inline uint64_t& work_queue_counter()
{
    static uint64_t c = 0;
    return c;
}

inline uint64_t push_work(std::function<void()> f,
                           std::shared_ptr<work_item>& out)
{
    std::lock_guard<std::mutex> lk(work_queue_mutex());
    uint64_t id = work_queue_counter()++;
    auto item   = std::make_shared<work_item>();
    item->func  = std::move(f);
    out         = item;
    work_queue_map()[id] = item;
    return id;
}

inline std::shared_ptr<work_item> pop_work(uint64_t id)
{
    std::lock_guard<std::mutex> lk(work_queue_mutex());
    auto it   = work_queue_map().find(id);
    auto item = it->second;
    work_queue_map().erase(it);
    return item;
}

// Legion task body executed by the runtime for each astm_async call.
inline void astm_worker_task_body(const Task* task,
                                  const std::vector<PhysicalRegion>&,
                                  Context ctx, Runtime* runtime)
{
    set_legion_context(runtime, ctx);
    uint64_t id = *reinterpret_cast<const uint64_t*>(task->args);
    auto item = pop_work(id);
    try {
        item->func();
        item->prom.set_value();
    } catch (...) {
        item->prom.set_exception(std::current_exception());
    }
}

// Call once before Runtime::start().
inline void register_astm_tasks()
{
    TaskVariantRegistrar registrar(ASTM_WORKER_TASK_ID, "astm_worker");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<astm_worker_task_body>(
        registrar, "astm_worker");
}

// Launch an arbitrary callable as a Legion child task; returns a future.
template <typename F, typename... Args>
inline legion_future<void> astm_async(F&& f, Args&&... args)
{
    auto bound = std::bind(std::forward<F>(f), std::forward<Args>(args)...);

    std::shared_ptr<work_item> item;
    uint64_t id = push_work([bound]() mutable { bound(); }, item);
    auto std_fut = item->prom.get_future();

    Runtime* rt  = get_legion_runtime();
    Context  ctx = get_legion_context();

    TaskLauncher launcher(ASTM_WORKER_TASK_ID,
                          TaskArgument(&id, sizeof(uint64_t)));
    rt->execute_task(ctx, launcher);        // child task – runtime tracks it

    return legion_future<void>(std::move(std_fut));
}

// =========================================================================
// Compatibility macros consumed by the .cpp test / driver files.
// =========================================================================
#define ASTM_MUTEX   std::mutex
#define ASTM_LOCK    std::unique_lock<std::mutex>
#define ASTM_FUTURE  astm::legion_future
#define ASTM_FUNCTION std::function
#define ASTM_ASYNC   astm::astm_async
#define ASTM_MAKE_READY_FUTURE astm::make_ready_future()

#define ASTM_TEST  assert
#define ASTM_REPORT 0

// =========================================================================
// shared_var_base
// =========================================================================
struct shared_var_base
{
    virtual ~shared_var_base() {}
    virtual shared_var_base* clone() const = 0;
    virtual void write(shared_var_base const&) = 0;
    virtual ASTM_LOCK lock() const = 0;
    virtual bool operator==(shared_var_base const&) const = 0;
};

// forward
struct transaction;

// =========================================================================
// transaction_future
// =========================================================================
struct transaction_future
{
    typedef legion_future<void> future_type;

private:
    transaction* trans_;
    future_type  fut_;

public:
    transaction_future(transaction* trans)
      : trans_(trans), fut_() {}

    transaction_future(transaction& trans)
      : trans_(&trans), fut_() {}

    template <typename F>
    void then(F f);

    void get() { fut_.get(); }
};

// =========================================================================
// shared_var<T>
// =========================================================================
template <typename T>
struct shared_var : shared_var_base
{
    typedef legion_future<void> future_type;

    // ----- local_var (transaction-local proxy) -----
    struct local_var
    {
    private:
        transaction*    trans_;
        shared_var_base* var_;

    public:
        local_var(transaction* trans, shared_var_base* var)
          : trans_(trans), var_(var) {}

        T get() const;
        operator T const& () const;
        local_var& operator=(shared_var_base const& rhs);
        local_var& operator=(T const& rhs);

        template <typename F>
        void then(F f);
    };

private:
    T                    data_;
    mutable std::mutex   mtx_;

    // Legion region backing
    LogicalRegion lr_;
    IndexSpace    is_;
    FieldSpace    fs_;
    bool          owns_region_;

    // Tag for lightweight clone construction (no region).
    struct clone_tag {};

    shared_var(T const& t, clone_tag)
      : data_(t)
      , owns_region_(false)
      , lr_(LogicalRegion::NO_REGION)
      , is_(IndexSpace::NO_SPACE)
      , fs_(FieldSpace::NO_SPACE)
    {}

    // Create the Legion region that mirrors data_.
    void init_region()
    {
        Runtime* rt = get_legion_runtime();
        if (rt) {
            Context ctx = get_legion_context();
            is_ = rt->create_index_space(ctx, Rect<1>(0, 0));
            fs_ = rt->create_field_space(ctx);
            {
                FieldAllocator fa = rt->create_field_allocator(ctx, fs_);
                fa.allocate_field(sizeof(T), ASTM_FID_VALUE);
            }
            lr_ = rt->create_logical_region(ctx, is_, fs_);
            sync_to_region();
        } else {
            lr_ = LogicalRegion::NO_REGION;
            is_ = IndexSpace::NO_SPACE;
            fs_ = FieldSpace::NO_SPACE;
            owns_region_ = false;
        }
    }

    // Push data_ → region.
    void sync_to_region()
    {
        Runtime* rt = get_legion_runtime();
        if (!rt || lr_ == LogicalRegion::NO_REGION) return;
        Context ctx = get_legion_context();

        InlineLauncher il(RegionRequirement(
            lr_, WRITE_DISCARD, EXCLUSIVE, lr_));
        il.add_field(ASTM_FID_VALUE);
        PhysicalRegion pr = rt->map_region(ctx, il);
        pr.wait_until_valid();

        const FieldAccessor<WRITE_DISCARD, T, 1> acc(pr, ASTM_FID_VALUE);
        acc[0] = data_;

        rt->unmap_region(ctx, pr);
    }

    // Pull region → data_  (used if needed for cross-task sync).
    void sync_from_region()
    {
        Runtime* rt = get_legion_runtime();
        if (!rt || lr_ == LogicalRegion::NO_REGION) return;
        Context ctx = get_legion_context();

        InlineLauncher il(RegionRequirement(
            lr_, READ_ONLY, EXCLUSIVE, lr_));
        il.add_field(ASTM_FID_VALUE);
        PhysicalRegion pr = rt->map_region(ctx, il);
        pr.wait_until_valid();

        const FieldAccessor<READ_ONLY, T, 1> acc(pr, ASTM_FID_VALUE);
        data_ = acc[0];

        rt->unmap_region(ctx, pr);
    }

public:
    future_type queue;

    shared_var()
      : data_(), mtx_(), owns_region_(true), queue()
    { init_region(); }

    shared_var(T const& t)
      : data_(t), mtx_(), owns_region_(true), queue()
    { init_region(); }

    shared_var(T&& t)
      : data_(std::move(t)), mtx_(), owns_region_(true), queue()
    { init_region(); }

    shared_var(shared_var const& rhs)
      : data_(rhs.data_), mtx_(), owns_region_(true), queue()
    { init_region(); }

    ~shared_var()
    {
        if (owns_region_ && lr_ != LogicalRegion::NO_REGION) {
            Runtime* rt = get_legion_runtime();
            if (rt) {
                Context ctx = get_legion_context();
                rt->destroy_logical_region(ctx, lr_);
                rt->destroy_field_space(ctx, fs_);
                rt->destroy_index_space(ctx, is_);
            }
        }
    }

    // Expose the region handle (for passing to child tasks).
    LogicalRegion get_region() const { return lr_; }

    // Locks, clones data_ (no region for clone).
    shared_var_base* clone() const
    {
        auto l = lock();
        return new shared_var(data_, clone_tag{});
    }

    // Fast read from local cache.
    T const& read() const { return data_; }

    // Write local cache and sync to region.
    void write(T const& rhs)
    {
        data_ = rhs;
        sync_to_region();
    }

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
};

// =========================================================================
// transaction
// =========================================================================
struct transaction
{
    std::list<
        std::pair<
            shared_var_base*,
            std::shared_ptr<shared_var_base>
        >
    > read_list;

    std::set<shared_var_base*> write_set;

    std::list<
        std::pair<
            legion_future<void>*,
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
        // 1.) Obtain exclusive access to all the variables.
        std::list<ASTM_LOCK> locks;
        for (auto const& var : variables)
        {
            assert(var.first != nullptr);
            locks.push_back((*var.first).lock());
        }

        // 2.) Compare recorded reads against current values.
        for (auto const& var : read_list)
        {
            assert(var.first != nullptr);
            if (!((*var.first) == (*var.second)))
            {
                clear();
                return false;           // conflict – retry
            }
        }

        // 3.) Perform writes from internal map → actual shared_vars.
        for (shared_var_base* var : write_set)
        {
            assert(var != nullptr);
            auto it = variables.find(var);
            assert(it != variables.end());
            (*var).write((*(*it).second));
        }

        // 4.) Perform async operations.
        for (auto& op : async_list)
        {
            if (op.first == nullptr)
                astm_async(op.second, this);      // fire-and-forget
            else
                (*op.first) = (*op.first).then(std::bind(op.second, this));
        }

        // 5.) Release exclusive access (RAII locks go out of scope).
        return true;
    }

    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != nullptr);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base>>
            entry(var, nullptr);

        auto result = variables.insert(entry);

        if (result.second)
        {
            // First read – clone the current value.
            (*result.first).second.reset((*var).clone());
            read_list.push_back(*result.first);
            return (*(*result.first).second);
        }
        else
            return (*(*result.first).second);
    }

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

    void then(legion_future<void>* fut,
              std::function<void(transaction*)> F)
    {
        async_list.push_back({fut, std::move(F)});
    }
};

// =========================================================================
// Out-of-line template definitions
// =========================================================================

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
    shared_var tmp(rhs, clone_tag{});
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
