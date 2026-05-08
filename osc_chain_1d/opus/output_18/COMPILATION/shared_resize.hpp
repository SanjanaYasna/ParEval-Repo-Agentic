// Copyright 2013 Mario Mulansky
// resizing functionality for odeint — Legion translation
#ifndef SHARED_RESIZE_HPP
#define SHARED_RESIZE_HPP

#include <iostream>
#include <cstddef>

#include <boost/numeric/odeint/util/state_wrapper.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/util/resize.hpp>
#include <boost/numeric/odeint/util/same_size.hpp>

#include "legion.h"

using namespace Legion;

// Field ID for the double values stored in regions
enum FieldIDs {
    FID_VAL = 100,
};

// The state_type represents a partitioned 1D array of doubles in Legion.
// It replaces the HPX vector< shared_future< shared_vec > >.
struct state_type {
    Runtime*         runtime;
    Context          ctx;
    LogicalRegion    region;
    IndexSpace       index_space;
    FieldSpace       field_space;
    IndexSpace       color_space;
    IndexPartition   index_partition;
    LogicalPartition logical_partition;
    size_t           num_blocks;   // number of partitions (M)
    size_t           block_size;   // elements per partition (G)
    size_t           total_size;   // total elements (N = M * G)
    bool             valid;

    state_type()
        : runtime(nullptr), ctx(),
          region(LogicalRegion::NO_REGION),
          index_space(IndexSpace::NO_SPACE),
          field_space(FieldSpace::NO_SPACE),
          color_space(IndexSpace::NO_SPACE),
          index_partition(IndexPartition::NO_PART),
          logical_partition(LogicalPartition::NO_PART),
          num_blocks(0), block_size(0), total_size(0), valid(false)
    {}

    size_t size() const { return num_blocks; }

    // Retrieve the i-th sub-region (replaces operator[] on the HPX vector)
    LogicalRegion get_subregion(size_t i) const {
        return runtime->get_logical_subregion_by_color(
            logical_partition,
            DomainPoint(Point<1>(static_cast<coord_t>(i))));
    }
};

// ---------------------------------------------------------------------------
// Helper: create a fully partitioned state (M blocks of G doubles each)
// ---------------------------------------------------------------------------
inline state_type create_state(Runtime* runtime, Context ctx,
                               size_t num_blocks, size_t block_size)
{
    state_type s;
    s.runtime    = runtime;
    s.ctx        = ctx;
    s.num_blocks = num_blocks;
    s.block_size = block_size;
    s.total_size = num_blocks * block_size;
    s.valid      = true;

    // 1-D index space [0, total_size)
    s.index_space = runtime->create_index_space(ctx,
        Rect<1>(Point<1>(0),
                Point<1>(static_cast<coord_t>(s.total_size) - 1)));

    // Single-field (double) field space
    s.field_space = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, s.field_space);
        fa.allocate_field(sizeof(double), FID_VAL);
    }

    // Logical region
    s.region = runtime->create_logical_region(ctx, s.index_space, s.field_space);

    // Equal partition into num_blocks contiguous blocks of block_size
    Rect<1> color_bounds(Point<1>(0),
                         Point<1>(static_cast<coord_t>(num_blocks) - 1));
    s.color_space = runtime->create_index_space(ctx, color_bounds);

    Transform<1, 1> transform;
    transform[0][0] = static_cast<coord_t>(block_size);
    Rect<1> extent(Point<1>(0),
                   Point<1>(static_cast<coord_t>(block_size) - 1));
    s.index_partition = runtime->create_partition_by_restriction(
        ctx, s.index_space, s.color_space, transform, extent);

    s.logical_partition =
        runtime->get_logical_partition(s.region, s.index_partition);

    return s;
}

// ---------------------------------------------------------------------------
// Helper: release Legion resources held by a state
// ---------------------------------------------------------------------------
inline void destroy_state(state_type &s)
{
    if (s.valid && s.runtime != nullptr) {
        s.runtime->destroy_logical_region(s.ctx, s.region);
        s.runtime->destroy_field_space(s.ctx, s.field_space);
        s.runtime->destroy_index_space(s.ctx, s.index_space);
        s.runtime->destroy_index_space(s.ctx, s.color_space);
        s.valid = false;
    }
}

// ===========================================================================
// boost::numeric::odeint specialisations for state_type
// ===========================================================================
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
    static bool same_size(const state_type &x1, const state_type &x2)
    {
        return (x1.num_blocks == x2.num_blocks) &&
               (x1.block_size == x2.block_size);
    }
};

template<>
struct resize_impl<state_type, state_type>
{
    static void resize(state_type &x1, const state_type &x2)
    {
        // Destroy previous Legion resources if present
        destroy_state(x1);
        // Allocate a new region + partition matching x2's shape
        x1 = create_state(x2.runtime, x2.ctx,
                          x2.num_blocks, x2.block_size);
    }
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif
