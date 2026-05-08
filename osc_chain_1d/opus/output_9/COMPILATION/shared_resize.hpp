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

// Field ID for double values stored in regions
enum FieldIDs {
    FID_VAL = 0,
};

// Global Legion runtime and context accessible from odeint callbacks
struct LegionHelper {
    static Runtime* runtime;
    static Context ctx;
};

// state_type: vector of LogicalRegions, each holding a block of G doubles
struct state_type {
    std::vector<LogicalRegion> blocks;
    size_t block_size; // G: number of elements per block

    state_type() : block_size(0) {}
    state_type(size_t n) : blocks(n), block_size(0) {}

    size_t size() const { return blocks.size(); }

    LogicalRegion& operator[](size_t i) { return blocks[i]; }
    const LogicalRegion& operator[](size_t i) const { return blocks[i]; }

    void resize(size_t n) { blocks.resize(n); }

    typedef std::vector<LogicalRegion>::iterator iterator;
    typedef std::vector<LogicalRegion>::const_iterator const_iterator;
    iterator begin() { return blocks.begin(); }
    iterator end() { return blocks.end(); }
    const_iterator begin() const { return blocks.begin(); }
    const_iterator end() const { return blocks.end(); }
};

// Helper: create a LogicalRegion of `size` doubles with field FID_VAL
inline LogicalRegion create_block_region(Runtime* runtime, Context ctx, size_t size)
{
    IndexSpace is = runtime->create_index_space(ctx,
        Rect<1>(0, static_cast<coord_t>(size) - 1));
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(double), FID_VAL);
    }
    return runtime->create_logical_region(ctx, is, fs);
}

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
        return (x1.size() == x2.size());
    }
};

template<>
struct resize_impl<state_type, state_type>
{
    static void resize(state_type &x1, const state_type &x2)
    {
        Runtime* runtime = LegionHelper::runtime;
        Context ctx = LegionHelper::ctx;

        x1.block_size = x2.block_size;
        x1.blocks.resize(x2.size());

        for (size_t i = 0; i < x2.size(); ++i)
        {
            x1.blocks[i] = create_block_region(runtime, ctx, x2.block_size);
            runtime->fill_field<double>(ctx, x1.blocks[i], x1.blocks[i],
                                        FID_VAL, 0.0);
        }
    }
};

} } }

#endif
