// Copyright 2013 Mario Mulansky
// resizing functionality for odeint — Legion version
#ifndef SHARED_RESIZE_HPP
#define SHARED_RESIZE_HPP

#include <iostream>
#include <cstddef>

#include <legion.h>

#include <boost/numeric/odeint/util/state_wrapper.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/util/resize.hpp>
#include <boost/numeric/odeint/util/same_size.hpp>

using namespace Legion;

// Field ID for the double values stored in each region element
enum FieldIDs {
    FID_VAL = 0,
};

// Replaces std::vector< shared_future< shared_vec > > from the HPX version.
// Wraps a partitioned LogicalRegion: total_size elements split into
// num_blocks sub-regions of block_size elements each.
struct state_type {
    LogicalRegion  lr;
    LogicalPartition lp;
    IndexSpace     is;
    IndexPartition ip;
    IndexSpace     color_is;
    FieldSpace     fs;

    size_t num_blocks;   // M – number of partitions / blocks
    size_t block_size;   // G – elements per block
    size_t total_size;   // N – total number of elements

    bool valid;
    bool owns_fs;        // whether this state owns (and should destroy) the FieldSpace

    Runtime* runtime;
    Context  ctx;

    // ----- construction / destruction -----

    state_type()
        : lr(LogicalRegion::NO_REGION),
          num_blocks(0), block_size(0), total_size(0),
          valid(false), owns_fs(false),
          runtime(nullptr)
    {}

    // Number of blocks (parallel to std::vector::size() in HPX version)
    size_t size() const { return num_blocks; }

    // Create a brand-new state with given dimensions
    void create(Runtime* rt, Context c, size_t N, size_t G)
    {
        runtime    = rt;
        ctx        = c;
        total_size = N;
        block_size = G;
        num_blocks = N / G;
        valid      = true;
        owns_fs    = true;

        // Index space [0, N-1]
        is = rt->create_index_space(ctx, Rect<1>(0, static_cast<coord_t>(N - 1)));

        // Field space – single double field
        fs = rt->create_field_space(ctx);
        {
            FieldAllocator fa = rt->create_field_allocator(ctx, fs);
            fa.allocate_field(sizeof(double), FID_VAL);
        }

        // Logical region
        lr = rt->create_logical_region(ctx, is, fs);

        // Equal partition into num_blocks pieces
        color_is = rt->create_index_space(ctx,
            Rect<1>(0, static_cast<coord_t>(num_blocks - 1)));
        ip = rt->create_equal_partition(ctx, is, color_is);
        lp = rt->get_logical_partition(ctx, lr, ip);
    }

    // Create a state with the same shape as `other` (reuses its FieldSpace)
    void create_like(const state_type& other)
    {
        if (valid) destroy();

        runtime    = other.runtime;
        ctx        = other.ctx;
        total_size = other.total_size;
        block_size = other.block_size;
        num_blocks = other.num_blocks;
        valid      = true;
        owns_fs    = false;        // borrow the FieldSpace from other
        fs         = other.fs;

        is = runtime->create_index_space(ctx,
            Rect<1>(0, static_cast<coord_t>(total_size - 1)));

        lr = runtime->create_logical_region(ctx, is, fs);

        color_is = runtime->create_index_space(ctx,
            Rect<1>(0, static_cast<coord_t>(num_blocks - 1)));
        ip = runtime->create_equal_partition(ctx, is, color_is);
        lp = runtime->get_logical_partition(ctx, lr, ip);
    }

    // Retrieve the i-th sub-region (replaces state_type[i] / futures[i])
    LogicalRegion get_subregion(size_t i) const
    {
        return runtime->get_logical_subregion_by_color(lp,
            DomainPoint(Point<1>(static_cast<coord_t>(i))));
    }

    // Release Legion resources
    void destroy()
    {
        if (valid && runtime) {
            runtime->destroy_logical_region(ctx, lr);
            runtime->destroy_index_space(ctx, color_is);
            runtime->destroy_index_space(ctx, is);
            if (owns_fs)
                runtime->destroy_field_space(ctx, fs);
        }
        valid   = false;
        owns_fs = false;
    }
};

// ---------------------------------------------------------------------------
// boost::numeric::odeint trait specialisations for state_type
// (mirror the HPX originals so the symplectic stepper can manage temporaries)
// ---------------------------------------------------------------------------

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
        return ( x1.total_size == x2.total_size ) &&
               ( x1.num_blocks == x2.num_blocks );
    }
};

template<>
struct resize_impl< state_type , state_type >
{
    static void resize( state_type &x1 ,
                        const state_type &x2 )
    {
        x1.create_like( x2 );
    }
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif
