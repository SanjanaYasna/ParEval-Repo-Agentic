// Copyright 2013 Mario Mulansky
// resizing functionality for odeint - Legion version
#ifndef SHARED_RESIZE_HPP
#define SHARED_RESIZE_HPP

#include <iostream>
#include <vector>

#include <boost/numeric/odeint/util/state_wrapper.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/util/resize.hpp>
#include <boost/numeric/odeint/util/same_size.hpp>

#include <legion.h>

using namespace Legion;

enum FieldIDs {
    FID_VAL = 0,
};

struct state_type {
    std::vector<LogicalRegion> regions;
    Runtime* runtime;
    Context ctx;
    size_t block_size;

    // Members used by system.hpp for partitioned access
    LogicalRegion lr;           // parent logical region
    LogicalPartition lp;        // logical partition
    size_t num_partitions;      // number of partitions
    size_t partition_size;      // size of each partition

    state_type() : runtime(nullptr), block_size(0),
                   lr(LogicalRegion::NO_REGION),
                   lp(LogicalPartition::NO_PART),
                   num_partitions(0), partition_size(0) {}

    size_t size() const { return regions.size(); }

    void resize(size_t n) { regions.resize(n); }

    LogicalRegion& operator[](size_t i) { return regions[i]; }
    const LogicalRegion& operator[](size_t i) const { return regions[i]; }

    // Range interface for boost::size compatibility
    auto begin() { return regions.begin(); }
    auto end() { return regions.end(); }
    auto begin() const { return regions.begin(); }
    auto end() const { return regions.end(); }
};

namespace boost {
namespace numeric {
namespace odeint {

template<>
struct is_resizeable<state_type>
{
    typedef boost::true_type type;
    const static bool value = type::value;
};

template<>
struct same_size_impl<state_type, state_type>
{
    static bool same_size(const state_type &x1,
                          const state_type &x2)
    {
        return (x1.size() == x2.size());
    }
};

template<>
struct resize_impl<state_type, state_type>
{
    static void resize(state_type &x1,
                       state_type x2)
    {
        x1.runtime = x2.runtime;
        x1.ctx = x2.ctx;
        x1.block_size = x2.block_size;
        x1.num_partitions = x2.num_partitions;
        x1.partition_size = x2.partition_size;
        x1.regions.resize(x2.size());

        // Create shared index and field spaces for all blocks
        IndexSpace is = x2.runtime->create_index_space(x2.ctx,
            Rect<1>(0, static_cast<coord_t>(x2.block_size) - 1));
        FieldSpace fs = x2.runtime->create_field_space(x2.ctx);
        {
            FieldAllocator allocator =
                x2.runtime->create_field_allocator(x2.ctx, fs);
            allocator.allocate_field(sizeof(double), FID_VAL);
        }

        for (size_t i = 0; i < x2.size(); ++i)
        {
            x1.regions[i] = x2.runtime->create_logical_region(x2.ctx, is, fs);
            // Initialize to zero, matching the HPX version's vector default
            x2.runtime->fill_field(x2.ctx, x1.regions[i],
                                   x1.regions[i], FID_VAL, 0.0);
        }
    }
};

} } }

#endif
