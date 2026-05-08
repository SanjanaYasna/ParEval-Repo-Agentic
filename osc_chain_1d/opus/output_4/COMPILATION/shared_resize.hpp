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
    FID_VAL = 101,
};

// Must be defined in odeint.cpp and set at the start of the top-level task
extern Context legion_ctx;
extern Runtime *legion_runtime;

struct state_type {
    std::vector<LogicalRegion> regions;
    size_t block_size;

    state_type() : block_size(0) {}

    size_t size() const { return regions.size(); }

    LogicalRegion& operator[](size_t i) { return regions[i]; }
    const LogicalRegion& operator[](size_t i) const { return regions[i]; }

    void resize(size_t n) { regions.resize(n); }

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
                       state_type x2)
    {
        x1.block_size = x2.block_size;
        x1.regions.resize(x2.size());
        for (size_t i = 0; i < x2.size(); ++i)
        {
            // Determine the block size from the source region's index space
            Domain dom = legion_runtime->get_index_space_domain(
                legion_ctx, x2.regions[i].get_index_space());
            Rect<1> rect = dom;
            size_t sz = rect.volume();

            // Create a new logical region of the same size
            IndexSpace is = legion_runtime->create_index_space(
                legion_ctx, Rect<1>(0, static_cast<coord_t>(sz) - 1));
            FieldSpace fs = legion_runtime->create_field_space(legion_ctx);
            {
                FieldAllocator fa = legion_runtime->create_field_allocator(
                    legion_ctx, fs);
                fa.allocate_field(sizeof(double), FID_VAL);
            }
            x1.regions[i] = legion_runtime->create_logical_region(
                legion_ctx, is, fs);
        }
    }
};

} } }

#endif
