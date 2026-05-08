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

// Set by the top-level task before any odeint operations
extern Runtime* legion_runtime;
extern Context legion_context;

struct state_type {
    std::vector<LogicalRegion> regions;
    size_t block_size; // G: number of elements per block

    state_type() : block_size(0) {}

    size_t size() const { return regions.size(); }

    LogicalRegion& operator[](size_t i) { return regions[i]; }
    const LogicalRegion& operator[](size_t i) const { return regions[i]; }

    typedef std::vector<LogicalRegion>::iterator iterator;
    typedef std::vector<LogicalRegion>::const_iterator const_iterator;

    iterator begin() { return regions.begin(); }
    iterator end() { return regions.end(); }
    const_iterator begin() const { return regions.begin(); }
    const_iterator end() const { return regions.end(); }
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
        return ( x1.size() == x2.size() );
    }
};

template<>
struct resize_impl< state_type , state_type >
{
    static void resize( state_type &x1 ,
                        state_type x2 )
    {
        const size_t M = x2.size();
        const size_t G = x2.block_size;

        x1.block_size = G;
        x1.regions.resize( M );

        // Create a shared index space for all blocks (all have size G)
        IndexSpace is = legion_runtime->create_index_space(
            legion_context,
            Rect<1>( 0, static_cast<coord_t>(G) - 1 ) );

        // Create a shared field space with a single double field
        FieldSpace fs = legion_runtime->create_field_space( legion_context );
        {
            FieldAllocator fa = legion_runtime->create_field_allocator(
                legion_context, fs );
            fa.allocate_field( sizeof(double), FID_VAL );
        }

        // Create one logical region per block
        for( size_t i = 0 ; i < M ; ++i )
        {
            x1.regions[i] = legion_runtime->create_logical_region(
                legion_context, is, fs );
        }
    }
};

} } }

#endif
