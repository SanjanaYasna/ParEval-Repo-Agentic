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

typedef std::vector<double> dvec;

enum FieldIDs {
    FID_VAL = 101,
};

// Global Legion context and runtime, set in the top-level task
extern Context legion_ctx;
extern Runtime *legion_runtime;

struct state_type {
    LogicalRegion lr;
    IndexSpace is;
    IndexSpace color_is;
    FieldSpace fs;
    IndexPartition ip;
    LogicalPartition lp;
    size_t num_blocks;
    size_t block_size;
    bool valid;

    state_type()
        : lr(LogicalRegion::NO_REGION),
          is(IndexSpace::NO_SPACE),
          color_is(IndexSpace::NO_SPACE),
          fs(FieldSpace::NO_SPACE),
          ip(IndexPartition::NO_PART),
          lp(LogicalPartition::NO_PART),
          num_blocks(0),
          block_size(0),
          valid(false)
    {}

    size_t size() const { return num_blocks; }

    LogicalRegion get_subregion(size_t i) const {
        return legion_runtime->get_logical_subregion_by_color(
            lp, DomainPoint(static_cast<coord_t>(i)));
    }

    void create(size_t nblocks, size_t bsize) {
        num_blocks = nblocks;
        block_size = bsize;
        size_t total = nblocks * bsize;

        // Create a 1-D index space covering all elements
        is = legion_runtime->create_index_space(legion_ctx,
            Rect<1>(0, static_cast<coord_t>(total - 1)));

        // Create field space with a single double field
        fs = legion_runtime->create_field_space(legion_ctx);
        {
            FieldAllocator fa =
                legion_runtime->create_field_allocator(legion_ctx, fs);
            fa.allocate_field(sizeof(double), FID_VAL);
        }

        // Create the logical region
        lr = legion_runtime->create_logical_region(legion_ctx, is, fs);

        // Partition into num_blocks equal-sized sub-regions
        color_is = legion_runtime->create_index_space(legion_ctx,
            Rect<1>(0, static_cast<coord_t>(num_blocks - 1)));
        ip = legion_runtime->create_equal_partition(legion_ctx, is, color_is);
        lp = legion_runtime->get_logical_partition(lr, ip);

        // Zero-initialize (analogous to std::vector<double> default init)
        legion_runtime->fill_field(legion_ctx, lr, lr, FID_VAL, 0.0);

        valid = true;
    }

    void destroy() {
        if (valid) {
            legion_runtime->destroy_logical_region(legion_ctx, lr);
            legion_runtime->destroy_field_space(legion_ctx, fs);
            legion_runtime->destroy_index_space(legion_ctx, color_is);
            legion_runtime->destroy_index_space(legion_ctx, is);
            valid = false;
        }
    }
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
        return (x1.num_blocks == x2.num_blocks) &&
               (x1.block_size == x2.block_size);
    }
};

template<>
struct resize_impl<state_type, state_type>
{
    static void resize(state_type &x1, state_type x2)
    {
        if (x1.valid) {
            x1.destroy();
        }
        x1.create(x2.num_blocks, x2.block_size);
    }
};

} } }

#endif
