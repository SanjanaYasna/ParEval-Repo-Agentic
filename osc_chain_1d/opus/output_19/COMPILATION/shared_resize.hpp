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

// The state_type wraps a logical region partitioned into M blocks of size G.
// This replaces the HPX vector of shared_future<shared_vec>.
struct state_type {
    LogicalRegion lr;
    IndexSpace is;
    FieldSpace fs;
    IndexPartition ip;
    LogicalPartition lp;
    IndexSpace color_is;
    std::vector<LogicalRegion> subregions;

    size_t num_blocks;   // M
    size_t block_size;   // G
    size_t total_size;   // N = M * G

    Runtime* runtime;
    Context ctx;
    bool valid;

    state_type()
        : lr(LogicalRegion::NO_REGION),
          num_blocks(0), block_size(0), total_size(0),
          runtime(nullptr), valid(false) {}

    size_t size() const { return num_blocks; }

    LogicalRegion& operator[](size_t i) { return subregions[i]; }
    const LogicalRegion& operator[](size_t i) const { return subregions[i]; }

    // Range interface for boost::size() compatibility
    auto begin() { return subregions.begin(); }
    auto end() { return subregions.end(); }
    auto begin() const { return subregions.begin(); }
    auto end() const { return subregions.end(); }

    void create(Runtime* rt, Context c, size_t M, size_t G) {
        runtime = rt;
        ctx = c;
        num_blocks = M;
        block_size = G;
        total_size = M * G;
        valid = true;

        // Create index space [0, N-1]
        Rect<1> rect(Point<1>(0), Point<1>((coord_t)(total_size - 1)));
        is = rt->create_index_space(c, rect);

        // Create field space with one double field
        fs = rt->create_field_space(c);
        {
            FieldAllocator fa = rt->create_field_allocator(c, fs);
            fa.allocate_field(sizeof(double), FID_VAL);
        }

        // Create the logical region
        lr = rt->create_logical_region(c, is, fs);

        // Partition into M equal blocks of size G
        Rect<1> color_rect(Point<1>(0), Point<1>((coord_t)(M - 1)));
        color_is = rt->create_index_space(c, color_rect);

        Transform<1, 1> transform;
        transform[0][0] = (coord_t)G;
        Rect<1> extent(Point<1>(0), Point<1>((coord_t)(G - 1)));
        ip = rt->create_partition_by_restriction(c, is, color_is, transform, extent);

        lp = rt->get_logical_partition(lr, ip);

        // Cache subregions for indexed access
        subregions.resize(M);
        for (size_t i = 0; i < M; i++) {
            subregions[i] = rt->get_logical_subregion_by_color(lp, (coord_t)i);
        }
    }

    void destroy() {
        if (valid && runtime != nullptr) {
            runtime->destroy_logical_region(ctx, lr);
            runtime->destroy_field_space(ctx, fs);
            runtime->destroy_index_space(ctx, color_is);
            runtime->destroy_index_space(ctx, is);
            subregions.clear();
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
    static void resize(state_type &x1, const state_type &x2)
    {
        if (x1.valid) {
            x1.destroy();
        }
        x1.create(x2.runtime, x2.ctx, x2.num_blocks, x2.block_size);
    }
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif
