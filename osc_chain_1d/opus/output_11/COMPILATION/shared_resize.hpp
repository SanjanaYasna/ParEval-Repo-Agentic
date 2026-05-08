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

#include "legion.h"

using namespace Legion;

enum FieldIDs {
    FID_VAL = 0,
};

struct state_type {
    LogicalRegion parent_region;
    IndexPartition ip;
    LogicalPartition lp;
    IndexSpace color_space;
    std::vector<LogicalRegion> sub_regions;
    Runtime* runtime;
    Context ctx;
    size_t block_size; // G: number of elements per block

    state_type()
        : parent_region(LogicalRegion::NO_REGION),
          runtime(nullptr), block_size(0)
    {}

    size_t size() const { return sub_regions.size(); }

    LogicalRegion& operator[](size_t i) { return sub_regions[i]; }
    const LogicalRegion& operator[](size_t i) const { return sub_regions[i]; }

    // Iterator support for boost::size() compatibility
    typedef std::vector<LogicalRegion>::iterator iterator;
    typedef std::vector<LogicalRegion>::const_iterator const_iterator;
    iterator begin() { return sub_regions.begin(); }
    iterator end() { return sub_regions.end(); }
    const_iterator begin() const { return sub_regions.begin(); }
    const_iterator end() const { return sub_regions.end(); }
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
        Runtime* runtime = x2.runtime;
        Context ctx = x2.ctx;
        size_t M = x2.size();
        size_t G = x2.block_size;

        // Create an index space for the full array of M*G elements
        Rect<1> elem_rect(0, static_cast<coord_t>(M * G) - 1);
        IndexSpace is = runtime->create_index_space(ctx, elem_rect);

        // Create a field space with one double-valued field
        FieldSpace fs = runtime->create_field_space(ctx);
        {
            FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
            fa.allocate_field(sizeof(double), FID_VAL);
        }

        // Create the logical region
        LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

        // Partition into M equal-sized blocks (each of size G)
        Rect<1> color_rect(0, static_cast<coord_t>(M) - 1);
        IndexSpace color_is = runtime->create_index_space(ctx, color_rect);
        IndexPartition ip_new = runtime->create_equal_partition(ctx, is, color_is);
        LogicalPartition lp_new = runtime->get_logical_partition(lr, ip_new);

        // Store into x1
        x1.parent_region = lr;
        x1.ip = ip_new;
        x1.lp = lp_new;
        x1.color_space = color_is;
        x1.runtime = runtime;
        x1.ctx = ctx;
        x1.block_size = G;

        x1.sub_regions.resize(M);
        for (size_t i = 0; i < M; ++i)
        {
            x1.sub_regions[i] = runtime->get_logical_subregion_by_color(
                lp_new, DomainPoint(Point<1>(static_cast<coord_t>(i))));
        }

        // Initialize all elements to zero
        // (matches HPX behavior where new vectors are value-initialized)
        runtime->fill_field(ctx, lr, lr, FID_VAL, 0.0);
    }
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif
