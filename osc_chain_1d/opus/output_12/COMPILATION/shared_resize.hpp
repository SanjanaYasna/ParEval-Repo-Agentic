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

// Global Legion runtime context, set once from the top-level task
struct LegionHelper {
    inline static Context ctx;
    inline static Runtime *runtime = nullptr;

    static void set(Context c, Runtime *r) {
        ctx = c;
        runtime = r;
    }
};

// state_type: a partitioned logical region representing a 1D array of doubles
// divided into M blocks of G elements each (analogous to
// std::vector< shared_future< shared_vec > > in the HPX version)
struct state_type {
    LogicalRegion lr;
    IndexSpace is;
    FieldSpace fs;
    IndexPartition ip;
    LogicalPartition lp;
    IndexSpace color_is;
    size_t M;     // number of partitions (blocks)
    size_t G;     // elements per block
    bool valid;

    state_type()
        : lr(LogicalRegion::NO_REGION), M(0), G(0), valid(false) {}

    state_type(size_t num_blocks, size_t block_size)
        : M(num_blocks), G(block_size), valid(true)
    {
        Context ctx = LegionHelper::ctx;
        Runtime *runtime = LegionHelper::runtime;
        size_t N = M * G;

        // Create index space for N elements
        is = runtime->create_index_space(ctx, Rect<1>(0, N - 1));

        // Create field space with a single double field
        fs = runtime->create_field_space(ctx);
        {
            FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
            fa.allocate_field(sizeof(double), FID_VAL);
        }

        // Create logical region
        lr = runtime->create_logical_region(ctx, is, fs);

        // Create equal partition into M blocks
        color_is = runtime->create_index_space(ctx, Rect<1>(0, M - 1));
        ip = runtime->create_equal_partition(ctx, is, color_is);
        lp = runtime->get_logical_partition(ctx, lr, ip);
    }

    size_t size() const { return M; }

    LogicalRegion subregion(size_t i) const {
        return LegionHelper::runtime->get_logical_subregion_by_color(lp, DomainPoint(Point<1>(i)));
    }

    void destroy() {
        if (valid && LegionHelper::runtime != nullptr) {
            Context ctx = LegionHelper::ctx;
            Runtime *runtime = LegionHelper::runtime;
            runtime->destroy_logical_region(ctx, lr);
            runtime->destroy_field_space(ctx, fs);
            runtime->destroy_index_space(ctx, is);
            runtime->destroy_index_space(ctx, color_is);
            valid = false;
            lr = LogicalRegion::NO_REGION;
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
        return (x1.M == x2.M) && (x1.G == x2.G);
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
        // Create new region with same block structure
        x1 = state_type(x2.M, x2.G);
        // Zero-initialize (original HPX code created zero-filled vectors)
        double zero = 0.0;
        LegionHelper::runtime->fill_field(
            LegionHelper::ctx, x1.lr, x1.lr, FID_VAL, zero);
    }
};

} } }

#endif
