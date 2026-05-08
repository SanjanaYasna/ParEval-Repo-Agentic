// Copyright 2013 Mario Mulansky
// resizing functionality for odeint - Legion version
#ifndef SHARED_RESIZE_HPP
#define SHARED_RESIZE_HPP

#include <iostream>
#include <vector>
#include <cassert>

#include <legion.h>

#include <boost/numeric/odeint/util/state_wrapper.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/util/resize.hpp>
#include <boost/numeric/odeint/util/same_size.hpp>

using namespace Legion;

// Field ID for storing double values in logical regions
enum FieldIDs {
    FID_VAL = 101,
};

// Global Legion runtime and context, set at top-level task entry
extern Runtime* legion_runtime;
extern Context legion_context;

// State type for Legion: a logical region partitioned into M blocks of G elements each.
// Replaces std::vector< shared_future< shared_vec > > from the HPX version.
struct state_type {
    LogicalRegion region;
    LogicalPartition partition;
    IndexSpace color_space;
    std::vector<LogicalRegion> subregions;
    size_t num_blocks;  // M
    size_t block_size;  // G

    state_type() : num_blocks(0), block_size(0) {}

    // Create the region and partition structure with M blocks of G elements
    void create(size_t M, size_t G) {
        num_blocks = M;
        block_size = G;
        size_t N = M * G;

        // Create index space [0, N-1]
        IndexSpace is = legion_runtime->create_index_space(
            legion_context, Rect<1>(0, N - 1));

        // Create field space with a single double field
        FieldSpace fs = legion_runtime->create_field_space(legion_context);
        {
            FieldAllocator fa = legion_runtime->create_field_allocator(
                legion_context, fs);
            fa.allocate_field(sizeof(double), FID_VAL);
        }

        // Create logical region
        region = legion_runtime->create_logical_region(legion_context, is, fs);

        // Create equal partition into M blocks
        color_space = legion_runtime->create_index_space(
            legion_context, Rect<1>(0, M - 1));
        IndexPartition ip = legion_runtime->create_equal_partition(
            legion_context, is, color_space);
        partition = legion_runtime->get_logical_partition(region, ip);

        // Cache sub-region handles for fast indexed access
        subregions.resize(M);
        for (size_t i = 0; i < M; ++i) {
            subregions[i] = legion_runtime->get_logical_subregion_by_color(
                partition, DomainPoint(Point<1>(i)));
        }
    }

    size_t size() const { return num_blocks; }

    LogicalRegion& operator[](size_t i) { return subregions[i]; }
    const LogicalRegion& operator[](size_t i) const { return subregions[i]; }

    // Iterator support for boost::size() and range-based operations
    typedef std::vector<LogicalRegion>::iterator iterator;
    typedef std::vector<LogicalRegion>::const_iterator const_iterator;
    iterator begin() { return subregions.begin(); }
    iterator end() { return subregions.end(); }
    const_iterator begin() const { return subregions.begin(); }
    const_iterator end() const { return subregions.end(); }
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
    static bool same_size(const state_type& x1, const state_type& x2)
    {
        return (x1.size() == x2.size());
    }
};

template<>
struct resize_impl<state_type, state_type>
{
    static void resize(state_type& x1, state_type x2)
    {
        // Create a new region with matching partition structure
        x1.create(x2.num_blocks, x2.block_size);
    }
};

} } }

#endif
