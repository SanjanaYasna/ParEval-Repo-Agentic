// Copyright 2013 Mario Mulansky
// resizing functionality for odeint — Legion translation
#ifndef SHARED_RESIZE_HPP
#define SHARED_RESIZE_HPP

#include <iostream>
#include <vector>
#include <memory>

#include <boost/numeric/odeint/util/state_wrapper.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/util/resize.hpp>
#include <boost/numeric/odeint/util/same_size.hpp>

#include "legion.h"

using namespace Legion;

typedef std::vector<double> dvec;

enum FieldIDs {
    FID_VAL = 101,
};

struct state_type {
    std::vector<LogicalRegion> regions;
    Runtime* runtime;
    Context ctx;
    size_t block_size; // number of doubles per block (G)

    state_type() : runtime(nullptr), block_size(0) {}

    size_t size() const { return regions.size(); }
    void resize(size_t n) { regions.resize(n); }

    LogicalRegion& operator[](size_t i) { return regions[i]; }
    const LogicalRegion& operator[](size_t i) const { return regions[i]; }

    // Range interface so that boost::size(state_type) works
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
                       const state_type &x2)
    {
        x1.runtime = x2.runtime;
        x1.ctx = x2.ctx;
        x1.block_size = x2.block_size;
        x1.regions.resize(x2.size());

        // Create a single shared field space for all blocks in this state
        FieldSpace fs = x1.runtime->create_field_space(x1.ctx);
        {
            FieldAllocator allocator =
                x1.runtime->create_field_allocator(x1.ctx, fs);
            allocator.allocate_field(sizeof(double), FID_VAL);
        }

        for (size_t i = 0; i < x2.size(); ++i)
        {
            // Each block gets its own index space (same bounds, unique handle)
            IndexSpace is = x1.runtime->create_index_space(x1.ctx,
                Rect<1>(0, static_cast<coord_t>(x1.block_size) - 1));
            x1.regions[i] = x1.runtime->create_logical_region(x1.ctx, is, fs);

            // Zero-initialize (matches std::vector<double>(n) behavior)
            x1.runtime->fill_field(x1.ctx, x1.regions[i],
                                   x1.regions[i], FID_VAL, 0.0);
        }
    }
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif
