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

typedef std::vector<double> dvec;

enum FieldIDs {
    FID_VAL = 0,
};

struct state_type {
    Runtime* runtime;
    Context ctx;
    std::vector<LogicalRegion> blocks;
    size_t block_size; // number of doubles per block (G)

    state_type() : runtime(nullptr), block_size(0) {}

    size_t size() const { return blocks.size(); }
    void resize(size_t n) { blocks.resize(n); }

    LogicalRegion& operator[](size_t i) { return blocks[i]; }
    const LogicalRegion& operator[](size_t i) const { return blocks[i]; }

    typedef std::vector<LogicalRegion>::iterator iterator;
    typedef std::vector<LogicalRegion>::const_iterator const_iterator;

    iterator begin() { return blocks.begin(); }
    iterator end() { return blocks.end(); }
    const_iterator begin() const { return blocks.begin(); }
    const_iterator end() const { return blocks.end(); }
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
        return (x1.size() == x2.size()) && (x1.block_size == x2.block_size);
    }
};

template<>
struct resize_impl<state_type, state_type>
{
    static void resize(state_type &x1, state_type x2)
    {
        x1.runtime = x2.runtime;
        x1.ctx = x2.ctx;
        x1.block_size = x2.block_size;
        x1.blocks.resize(x2.size());

        for (size_t i = 0; i < x2.size(); ++i)
        {
            // Create an index space matching the block size
            IndexSpace is = x2.runtime->create_index_space(
                x2.ctx, Rect<1>(0, (coord_t)(x2.block_size - 1)));
            FieldSpace fs = x2.runtime->create_field_space(x2.ctx);
            {
                FieldAllocator fa =
                    x2.runtime->create_field_allocator(x2.ctx, fs);
                fa.allocate_field(sizeof(double), FID_VAL);
            }
            x1.blocks[i] = x2.runtime->create_logical_region(
                x2.ctx, is, fs);
            // Zero-initialize (mirrors HPX version creating zero-filled vectors)
            x2.runtime->fill_field(
                x2.ctx, x1.blocks[i], x1.blocks[i], FID_VAL, 0.0);
        }
    }
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif
