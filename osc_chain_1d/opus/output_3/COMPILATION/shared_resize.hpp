// Copyright 2013 Mario Mulansky
// resizing functionality for odeint – Legion translation
#ifndef SHARED_RESIZE_HPP
#define SHARED_RESIZE_HPP

#include <iostream>
#include <vector>

#include <boost/numeric/odeint/util/state_wrapper.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/util/resize.hpp>
#include <boost/numeric/odeint/util/same_size.hpp>

#include "legion.h"

using namespace Legion;

enum FieldIDs {
    FID_VAL = 101,
};

struct state_type {
    Runtime *runtime;
    Context ctx;
    LogicalRegion region;
    LogicalPartition lp;
    IndexSpace color_space;
    std::vector<LogicalRegion> subregions;
    size_t num_partitions;  // M (number of blocks)
    size_t partition_size;  // G (elements per block)

    state_type()
        : runtime(nullptr), num_partitions(0), partition_size(0)
    {}

    state_type(Runtime *rt, Context c, size_t M, size_t G)
        : runtime(rt), ctx(c), num_partitions(0), partition_size(0)
    {
        init(M, G);
    }

    void init(size_t M, size_t G)
    {
        num_partitions = M;
        partition_size = G;
        size_t N = M * G;

        IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, N - 1));
        FieldSpace fs = runtime->create_field_space(ctx);
        {
            FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
            fa.allocate_field(sizeof(double), FID_VAL);
        }
        region = runtime->create_logical_region(ctx, is, fs);

        color_space = runtime->create_index_space(ctx, Rect<1>(0, M - 1));
        IndexPartition ip = runtime->create_equal_partition(ctx, is, color_space);
        lp = runtime->get_logical_partition(ctx, region, ip);

        subregions.resize(M);
        for (size_t i = 0; i < M; i++)
        {
            subregions[i] = runtime->get_logical_subregion_by_color(ctx, lp, i);
        }
    }

    size_t size() const { return num_partitions; }

    LogicalRegion& operator[](size_t i) { return subregions[i]; }
    const LogicalRegion& operator[](size_t i) const { return subregions[i]; }

    // Range interface so boost::size(state_type) works
    auto begin() { return subregions.begin(); }
    auto end() { return subregions.end(); }
    auto begin() const { return subregions.begin(); }
    auto end() const { return subregions.end(); }
};

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
        return ( x1.num_partitions == x2.num_partitions ) &&
               ( x1.partition_size == x2.partition_size );
    }
};

template<>
struct resize_impl< state_type , state_type >
{
    static void resize( state_type &x1 ,
                        const state_type &x2 )
    {
        x1.runtime = x2.runtime;
        x1.ctx = x2.ctx;
        x1.init( x2.num_partitions , x2.partition_size );
    }
};

} } }

#endif
