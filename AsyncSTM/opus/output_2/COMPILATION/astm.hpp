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

#include <list>
#include <set>
#include <map>
#include <vector>
#include <functional>
#include <memory>
#include <cassert>

namespace astm
{

// ------------------------------------------------------------------
// Legion runtime / context helpers.
// Call init_context() at the start of every Legion task body.
// ------------------------------------------------------------------
inline Legion::Runtime*& legion_runtime_ref()
{
    static thread_local Legion::Runtime* r = nullptr;
    return r;
}

inline Legion::Context& legion_ctx_ref()
{
    static thread_local Legion::Context c;
    return c;
}

inline void init_context(Legion::Runtime* rt, Legion::Context c)
{
    legion_runtime_ref() = rt;
    legion_ctx_ref()     = c;
}

inline Legion::Runtime* lg_rt()  { return legion_runtime_ref(); }
inline Legion::Context  lg_ctx() { return legion_ctx_ref(); }

// ------------------------------------------------------------------
// Field ID used by every shared_var region
// ------------------------------------------------------------------
enum { FID_ASTM_DATA = 200 };

// ------------------------------------------------------------------
// shared_var_base – type-erased interface
// ------------------------------------------------------------------
struct shared_var_base
{
    virtual ~shared_var_base() {}

    virtual shared_var_base* clone() const = 0;

    virtual void write(shared_var_base const&) = 0;

    virtual bool operator==(shared_var_base const&) const = 0;

    // Legion helpers used by transaction::commit_transaction
    virtual Legion::LogicalRegion logical_region() const = 0;
    virtual bool  has_region() const = 0;
    virtual void  sync_from_mapped(Legion::PhysicalRegion const& pr) = 0;
    virtual void  sync_to_mapped  (Legion::PhysicalRegion const& pr) = 0;
};

struct transaction;

// ------------------------------------------------------------------
// transaction_future
// ------------------------------------------------------------------
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

    // Callbacks are executed synchronously during commit, so get() is a no-op.
    void get() {}
};

// ------------------------------------------------------------------
// shared_var<T>
// ------------------------------------------------------------------
template <typename T>
struct shared_var : shared_var_base
{
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
    mutable T data_;
    Legion::LogicalRegion lr_;
    Legion::IndexSpace    is_;
    Legion::FieldSpace    fs_;
    bool owns_region_;

    // Tag type – selects the private constructor that skips region creation.
    struct no_region_tag {};

    // Clone constructor: in-memory only, no Legion region.
    shared_var(T const& t, no_region_tag)
      : data_(t)
      , lr_(Legion::LogicalRegion::NO_REGION)
      , is_(Legion::IndexSpace::NO_SPACE)
      , fs_(Legion::FieldSpace::NO_SPACE)
      , owns_region_(false)
    {}

    // Helper: create a single-element logical region.
    void create_region()
    {
        Legion::Runtime* runtime = lg_rt();
        Legion::Context  c       = lg_ctx();

        Legion::Rect<1> bounds(0, 0);
        is_ = runtime->create_index_space(c, bounds);
        fs_ = runtime->create_field_space(c);
        {
            Legion::FieldAllocator fa =
                runtime->create_field_allocator(c, fs_);
            fa.allocate_field(sizeof(T), FID_ASTM_DATA);
        }
        lr_ = runtime->create_logical_region(c, is_, fs_);
        owns_region_ = true;
    }

    // Helper: write a value to the region through an inline mapping.
    void write_to_region(T const& val) const
    {
        Legion::Runtime* runtime = lg_rt();
        Legion::Context  c       = lg_ctx();

        Legion::InlineLauncher il(
            Legion::RegionRequirement(lr_, WRITE_DISCARD, EXCLUSIVE, lr_));
        il.add_field(FID_ASTM_DATA);

        Legion::PhysicalRegion pr = runtime->map_region(c, il);
        pr.wait_until_valid();

        const Legion::FieldAccessor<WRITE_DISCARD, T, 1>
            acc(pr, FID_ASTM_DATA);
        acc[0] = val;

        runtime->unmap_region(c, pr);
    }

    // Helper: read the value from the region through an inline mapping.
    T read_from_region() const
    {
        Legion::Runtime* runtime = lg_rt();
        Legion::Context  c       = lg_ctx();

        Legion::InlineLauncher il(
            Legion::RegionRequirement(lr_, READ_ONLY, EXCLUSIVE, lr_));
        il.add_field(FID_ASTM_DATA);

        Legion::PhysicalRegion pr = runtime->map_region(c, il);
        pr.wait_until_valid();

        const Legion::FieldAccessor<READ_ONLY, T, 1>
            acc(pr, FID_ASTM_DATA);
        T val = acc[0];

        runtime->unmap_region(c, pr);
        return val;
    }

  public:
    shared_var()
      : data_(), owns_region_(false)
    {
        create_region();
        write_to_region(data_);
    }

    shared_var(T const& t)
      : data_(t), owns_region_(false)
    {
        create_region();
        write_to_region(data_);
    }

    shared_var(T&& t)
      : data_(std::move(t)), owns_region_(false)
    {
        create_region();
        write_to_region(data_);
    }

    shared_var(shared_var const& rhs)
      : data_(rhs.data_), owns_region_(false)
    {
        create_region();
        write_to_region(data_);
    }

    ~shared_var()
    {
        if (owns_region_ && lg_rt() != nullptr)
        {
            Legion::Context c = lg_ctx();
            lg_rt()->destroy_logical_region(c, lr_);
            lg_rt()->destroy_field_space(c, fs_);
            lg_rt()->destroy_index_space(c, is_);
        }
    }

    // ----------------------------------------------------------
    // shared_var_base interface
    // ----------------------------------------------------------

    // Reads the current region value and returns a lightweight in-memory copy.
    shared_var_base* clone() const override
    {
        if (owns_region_)
        {
            T val = read_from_region();
            return new shared_var(val, no_region_tag{});
        }
        return new shared_var(data_, no_region_tag{});
    }

    // In-memory write (used by transaction commit to set data_ before
    // flushing to region).
    void write(shared_var_base const& rhs) override
    {
        data_ = dynamic_cast<shared_var const*>(&rhs)->data_;
    }

    bool operator==(shared_var_base const& rhs) const override
    {
        return data_ == dynamic_cast<shared_var const*>(&rhs)->data_;
    }

    Legion::LogicalRegion logical_region() const override { return lr_; }
    bool has_region() const override { return owns_region_; }

    // Read data_ from an already-mapped physical region.
    void sync_from_mapped(Legion::PhysicalRegion const& pr) override
    {
        const Legion::FieldAccessor<READ_WRITE, T, 1>
            acc(pr, FID_ASTM_DATA);
        data_ = acc[0];
    }

    // Write data_ into an already-mapped physical region.
    void sync_to_mapped(Legion::PhysicalRegion const& pr) override
    {
        const Legion::FieldAccessor<READ_WRITE, T, 1>
            acc(pr, FID_ASTM_DATA);
        acc[0] = data_;
    }

    // ----------------------------------------------------------
    // Public helpers used directly by application code
    // ----------------------------------------------------------

    // Read with automatic region sync.
    T const& read() const
    {
        if (owns_region_)
            data_ = read_from_region();
        return data_;
    }

    // Direct write (outside any transaction).
    void write(T const& rhs)
    {
        data_ = rhs;
        if (owns_region_)
            write_to_region(rhs);
    }

    local_var get_local(transaction& trans)
    {
        return local_var(&trans, this);
    }
};

// ------------------------------------------------------------------
// transaction
// ------------------------------------------------------------------
struct transaction
{
    std::list<
        std::pair<
            shared_var_base*                    // the shared variable
          , std::shared_ptr<shared_var_base>    // value at read time
        >
    > read_list;

    std::set<
        shared_var_base*                        // variables written
    > write_set;

    std::list<
        std::pair<
            void*                                       // (unused, kept for API compat)
          , std::function<void(transaction*)>            // callback
        >
    > async_list;

    std::map<
        shared_var_base*                        // shared variable
      , std::shared_ptr<shared_var_base>        // current transaction-local value
    > variables;

    void clear()
    {
        read_list.clear();
        write_set.clear();
        async_list.clear();
        variables.clear();
    }

    // Commit using Legion inline mappings for exclusive access.
    bool commit_transaction()
    {
        // Algorithm:
        //
        // 1.) Inline-map all involved regions for READ_WRITE (exclusive).
        // 2.) Sync cached data_ from mapped regions.
        // 3.) Verify recorded reads against current values (fail if needed).
        // 4.) Perform writes (to data_).
        // 5.) Flush written data_ to mapped regions.
        // 6.) Execute async callbacks.
        // 7.) Unmap all regions (release exclusive access).

        Legion::Runtime* runtime = lg_rt();
        Legion::Context  c       = lg_ctx();

        // 1.) Inline-map all involved regions.
        //     The std::map is sorted by pointer, giving a consistent
        //     mapping order (analogous to the original lock order).
        std::map<shared_var_base*, Legion::PhysicalRegion> mapped;

        for (auto const& entry : variables)
        {
            assert(entry.first != nullptr);

            if (entry.first->has_region())
            {
                Legion::LogicalRegion lr = entry.first->logical_region();
                Legion::InlineLauncher il(
                    Legion::RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
                il.add_field(FID_ASTM_DATA);

                Legion::PhysicalRegion pr = runtime->map_region(c, il);
                pr.wait_until_valid();
                mapped[entry.first] = pr;
            }
        }

        // 2.) Sync data_ from the mapped regions so that the virtual
        //     operator== sees the actual current values.
        for (auto& m : mapped)
            m.first->sync_from_mapped(m.second);

        // 3.) Verify recorded reads.
        for (auto const& entry : read_list)
        {
            assert(entry.first != nullptr);

            if (!((*entry.first) == (*entry.second)))
            {
                // Conflict – unmap everything, clear state, signal retry.
                for (auto& m : mapped)
                    runtime->unmap_region(c, m.second);

                clear();
                return false;
            }
        }

        // 4.) Perform writes (in-memory).
        for (shared_var_base* var : write_set)
        {
            assert(var != nullptr);

            auto it = variables.find(var);
            assert(it != variables.end());

            (*var).write((*(*it).second));
        }

        // 5.) Flush written values to their regions.
        for (shared_var_base* var : write_set)
        {
            auto mit = mapped.find(var);
            if (mit != mapped.end())
                var->sync_to_mapped(mit->second);
        }

        // 6.) Execute async callbacks synchronously (regions still mapped,
        //     so exclusive access is maintained).
        for (auto& op : async_list)
            op.second(this);

        // 7.) Unmap all regions.
        for (auto& m : mapped)
            runtime->unmap_region(c, m.second);

        return true;
    }

    shared_var_base const& read(shared_var_base* var)
    {
        assert(var != nullptr);

        std::pair<shared_var_base*, std::shared_ptr<shared_var_base>>
            entry(var, std::shared_ptr<shared_var_base>());

        auto result = variables.insert(entry);

        if (result.second == true)
        {
            // First access – clone current value from the region.
            (*result.first).second.reset((*var).clone());

            // Record the read.
            read_list.push_back(*result.first);

            return (*(*result.first).second);
        }
        else
            // Already in transaction state.
            return (*(*result.first).second);
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

// ------------------------------------------------------------------
// local_var out-of-line definitions
// ------------------------------------------------------------------
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
    shared_var tmp(rhs, no_region_tag{});
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
