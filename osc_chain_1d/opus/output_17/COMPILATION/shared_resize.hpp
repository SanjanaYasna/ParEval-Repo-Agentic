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

struct state_type {
    std::vector<LogicalRegion> partitions;

    // Static references to Legion runtime context, set before odeint calls
    static Runtime* rt;
    static Context ctx;
    static FieldSpace fs;

    size_t size() const { return partitions.size(); }
    void resize(size_t n) { partitions.resize(n); }

    LogicalRegion& operator[](size_t i) { return partitions[i]; }
    const LogicalRegion& operator[](size_t i) const { return partitions[i]; }

    // Boost.Range / iteration support (needed by boost::size in algebra)
    typedef std::vector<LogicalRegion>::iterator iterator;
    typedef std::vector<LogicalRegion>::const_iterator const_iterator;
    iterator begin() { return partitions.begin(); }
    iterator end() { return partitions.end(); }
    const_iterator begin() const { return partitions.begin(); }
    const_iterator end() const { return partitions.end(); }
};

// C++17 inline static member definitions
inline Runtime* state_type::rt = nullptr;
inline Context state_type::ctx = {};
inline FieldSpace state_type::fs = {};

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
        Runtime* runtime = state_type::rt;
        Context ctx = state_type::ctx;
        FieldSpace fs = state_type::fs;

        x1.resize(x2.size());
        for (size_t i = 0; i < x2.size(); ++i)
        {
            // Determine the size of the source block's index space
            IndexSpace is_src = x2[i].get_index_space();
            Domain domain = runtime->get_index_space_domain(ctx, is_src);
            Rect<1> rect = domain;

            // Create a new index space and logical region of the same size
            IndexSpace is_new = runtime->create_index_space(ctx, rect);
            x1[i] = runtime->create_logical_region(ctx, is_new, fs);

            // Zero-initialize (matches HPX version's default-constructed vector)
            runtime->fill_field(ctx, x1[i], x1[i], FID_VAL, 0.0);
        }
    }
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif
