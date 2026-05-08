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

// Field ID for the double values stored in each block region
enum FieldIDs {
    FID_VAL = 101,
};

// Global pointers to Legion runtime and context, to be set in the top-level task
extern Runtime *legion_runtime;
extern Context legion_context;

// state_type: a collection of M LogicalRegions, each representing one block of G doubles
struct state_type {
    std::vector<LogicalRegion> regions;
    size_t block_size; // number of elements per block (G)

    state_type() : block_size(0) {}

    size_t size() const { return regions.size(); }

    void resize(size_t n) {
        regions.resize(n);
    }

    LogicalRegion& operator[](size_t i) { return regions[i]; }
    const LogicalRegion& operator[](size_t i) const { return regions[i]; }

    // Range interface for boost::size / boost::range compatibility
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
        return (x1.size() == x2.size()) && (x1.block_size == x2.block_size);
    }
};

template<>
struct resize_impl<state_type, state_type>
{
    static void resize(state_type &x1, state_type x2)
    {
        x1.resize(x2.size());
        x1.block_size = x2.block_size;

        for (size_t i = 0; i < x2.size(); ++i)
        {
            // Create an index space with block_size points [0, block_size-1]
            IndexSpace is = legion_runtime->create_index_space(
                legion_context,
                Rect<1>(0, static_cast<coord_t>(x2.block_size) - 1));

            // Create a field space with a single double field
            FieldSpace fs = legion_runtime->create_field_space(legion_context);
            {
                FieldAllocator fa = legion_runtime->create_field_allocator(
                    legion_context, fs);
                fa.allocate_field(sizeof(double), FID_VAL);
            }

            // Create the logical region for this block
            x1.regions[i] = legion_runtime->create_logical_region(
                legion_context, is, fs);

            // Zero-initialize the region
            legion_runtime->fill_field(
                legion_context, x1.regions[i], x1.regions[i],
                FID_VAL, 0.0);
        }
    }
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif
