// Copyright 2013 Mario Mulansky
// resizing functionality for odeint - Legion version
#ifndef SHARED_RESIZE_HPP
#define SHARED_RESIZE_HPP

#include <iostream>
#include <vector>

#include <legion.h>

#include <boost/numeric/odeint/util/state_wrapper.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/util/resize.hpp>
#include <boost/numeric/odeint/util/same_size.hpp>

using namespace Legion;

enum FieldIDs {
    FID_VAL = 0,
};

struct state_type {
    LogicalRegion parent_lr;
    LogicalPartition lp;
    IndexSpace color_space;
    std::vector<LogicalRegion> subregions;
    size_t M;  // number of partitions (blocks)
    size_t G;  // number of elements per partition (block size)
    Context ctx;
    Runtime *runtime;
    bool valid;

    state_type() : M(0), G(0), runtime(nullptr), valid(false) {}

    size_t size() const { return M; }

    LogicalRegion& operator[](size_t i) { return subregions[i]; }
    const LogicalRegion& operator[](size_t i) const { return subregions[i]; }

    // Range interface for boost::size compatibility
    auto begin() { return subregions.begin(); }
    auto end() { return subregions.end(); }
    auto begin() const { return subregions.begin(); }
    auto end() const { return subregions.end(); }
};

// Helper function to create and initialize a partitioned state
inline state_type create_state(Context ctx, Runtime *runtime, size_t M, size_t G)
{
    state_type s;
    s.M = M;
    s.G = G;
    s.ctx = ctx;
    s.runtime = runtime;
    s.valid = true;

    // Create index space for all M*G elements
    Rect<1> elem_rect(0, static_cast<coord_t>(M * G) - 1);
    IndexSpace is = runtime->create_index_space(ctx, elem_rect);

    // Create field space with a single double-valued field
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(double), FID_VAL);
    }

    // Create the parent logical region
    s.parent_lr = runtime->create_logical_region(ctx, is, fs);

    // Create color space for M partitions
    Rect<1> color_rect(0, static_cast<coord_t>(M) - 1);
    s.color_space = runtime->create_index_space(ctx, color_rect);

    // Create equal partition (each partition gets G elements)
    IndexPartition ip = runtime->create_equal_partition(ctx, is, s.color_space);
    s.lp = runtime->get_logical_partition(ctx, s.parent_lr, ip);

    // Cache subregion handles for indexed access
    s.subregions.resize(M);
    for (size_t i = 0; i < M; ++i) {
        s.subregions[i] = runtime->get_logical_subregion_by_color(
            s.lp, DomainPoint(Point<1>(static_cast<coord_t>(i))));
    }

    // Initialize all elements to zero
    runtime->fill_field(ctx, s.parent_lr, s.parent_lr, FID_VAL, 0.0);

    return s;
}

namespace boost {
namespace numeric {
namespace odeint {

template<>
struct is_resizeable< state_type >
{
    typedef boost::true_type type;
    const static bool value = type::value;
};

template<>
struct same_size_impl< state_type , state_type >
{
    static bool same_size( const state_type &x1 ,
                           const state_type &x2 )
    {
        return ( x1.M == x2.M ) && ( x1.G == x2.G );
    }
};

template<>
struct resize_impl< state_type , state_type >
{
    static void resize( state_type &x1 ,
                        state_type x2 )
    {
        x1 = create_state( x2.ctx , x2.runtime , x2.M , x2.G );
    }
};

} } }

#endif
