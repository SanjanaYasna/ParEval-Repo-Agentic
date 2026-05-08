// Copyright 2013 Mario Mulansky
// resizing functionality for odeint — Legion translation
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
    std::vector<LogicalRegion> regions;
    Runtime *runtime;
    Context ctx;

    state_type() : runtime(nullptr), ctx() {}

    explicit state_type(size_t n) : regions(n), runtime(nullptr), ctx() {}

    size_t size() const { return regions.size(); }

    LogicalRegion& operator[](size_t i) { return regions[i]; }
    const LogicalRegion& operator[](size_t i) const { return regions[i]; }

    void resize(size_t n) { regions.resize(n); }

    // Iterators for boost::size / range compatibility
    auto begin() { return regions.begin(); }
    auto end() { return regions.end(); }
    auto begin() const { return regions.begin(); }
    auto end() const { return regions.end(); }
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
        x1.runtime = x2.runtime;
        x1.ctx = x2.ctx;
        x1.regions.resize(x2.size());

        for (size_t i = 0; i < x2.size(); ++i)
        {
            // Determine the size of x2's i-th region
            Domain dom = x1.runtime->get_index_space_domain(
                x2.regions[i].get_index_space());
            Rect<1> rect = dom;
            size_t n = rect.hi[0] - rect.lo[0] + 1;

            // Create a new logical region with the same size
            IndexSpace is = x1.runtime->create_index_space(
                x1.ctx, Rect<1>(0, n - 1));
            FieldSpace fs = x1.runtime->create_field_space(x1.ctx);
            {
                FieldAllocator fa = x1.runtime->create_field_allocator(
                    x1.ctx, fs);
                fa.allocate_field(sizeof(double), FID_VAL);
            }
            x1.regions[i] = x1.runtime->create_logical_region(
                x1.ctx, is, fs);

            // Zero-initialize the new region (matches HPX behavior
            // where dvec(size) value-initializes to 0.0)
            x1.runtime->fill_field(x1.ctx, x1.regions[i],
                                   x1.regions[i], FID_VAL, 0.0);
        }
    }
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif
