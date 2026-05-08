// Copyright 2013 Mario Mulansky
// resizing functionality for odeint (Legion version)
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
    Runtime* runtime;
    Context ctx;
    std::vector<LogicalRegion> regions;
    size_t block_size;

    state_type() : runtime(nullptr), block_size(0) {}

    size_t size() const { return regions.size(); }

    void resize(size_t n) { regions.resize(n); }

    LogicalRegion& operator[](size_t i) { return regions[i]; }
    const LogicalRegion& operator[](size_t i) const { return regions[i]; }

    // For boost::size() / range compatibility
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
        x1.runtime = x2.runtime;
        x1.ctx = x2.ctx;
        x1.block_size = x2.block_size;
        x1.regions.resize( x2.size() );

        for( size_t i = 0 ; i < x2.size() ; ++i )
        {
            // Query the size of the source region's index space
            Domain dom = x2.runtime->get_index_space_domain(
                x2.regions[i].get_index_space() );
            Rect<1> rect = dom;
            size_t region_size = rect.hi[0] - rect.lo[0] + 1;

            // Create a new logical region with the same size
            IndexSpace is = x2.runtime->create_index_space( x2.ctx ,
                Rect<1>( 0, static_cast<coord_t>(region_size) - 1 ) );
            FieldSpace fs = x2.runtime->create_field_space( x2.ctx );
            {
                FieldAllocator allocator =
                    x2.runtime->create_field_allocator( x2.ctx , fs );
                allocator.allocate_field( sizeof(double) , FID_VAL );
            }
            x1.regions[i] = x2.runtime->create_logical_region( x2.ctx , is , fs );

            // Zero-initialize the new region (deferred, like the HPX dataflow version)
            x2.runtime->fill_field( x2.ctx , x1.regions[i] ,
                                    x1.regions[i] , FID_VAL , 0.0 );
        }
    }
};

} } }

#endif
