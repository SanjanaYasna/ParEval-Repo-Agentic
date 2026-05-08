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

// Global runtime and context, to be set by the top-level task
extern Runtime* legion_runtime;
extern Context  legion_context;

// state_type for Legion: a container of LogicalRegions, each holding one block of doubles
struct state_type {
    std::vector<LogicalRegion> blocks;
    std::vector<size_t>        block_sizes;

    // Container-like interface for boost::size and odeint compatibility
    typedef std::vector<LogicalRegion>::iterator       iterator;
    typedef std::vector<LogicalRegion>::const_iterator  const_iterator;
    typedef LogicalRegion  value_type;
    typedef LogicalRegion& reference;
    typedef const LogicalRegion& const_reference;
    typedef size_t         size_type;

    state_type() = default;
    explicit state_type(size_t n) : blocks(n), block_sizes(n, 0) {}

    size_t size() const { return blocks.size(); }
    bool   empty() const { return blocks.empty(); }

    iterator       begin()       { return blocks.begin(); }
    iterator       end()         { return blocks.end(); }
    const_iterator begin() const { return blocks.begin(); }
    const_iterator end()   const { return blocks.end(); }

    void resize(size_t n) {
        blocks.resize(n);
        block_sizes.resize(n, 0);
    }

    LogicalRegion&       operator[](size_t i)       { return blocks[i]; }
    const LogicalRegion& operator[](size_t i) const { return blocks[i]; }
};

// Helper: create a logical region for one block of the given size
inline LogicalRegion create_block_region(size_t block_size) {
    Rect<1> rect(0, static_cast<coord_t>(block_size) - 1);
    IndexSpace is = legion_runtime->create_index_space(legion_context, rect);
    FieldSpace fs = legion_runtime->create_field_space(legion_context);
    {
        FieldAllocator fa = legion_runtime->create_field_allocator(legion_context, fs);
        fa.allocate_field(sizeof(double), FID_VAL);
    }
    return legion_runtime->create_logical_region(legion_context, is, fs);
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
    static bool same_size(const state_type &x1,
                          const state_type &x2)
    {
        if (x1.size() != x2.size()) return false;
        for (size_t i = 0; i < x1.size(); ++i) {
            if (x1.block_sizes[i] != x2.block_sizes[i]) return false;
        }
        return true;
    }
};

template<>
struct resize_impl<state_type, state_type>
{
    static void resize(state_type &x1,
                       const state_type &x2)
    {
        x1.resize(x2.size());
        for (size_t i = 0; i < x2.size(); ++i)
        {
            size_t bs = x2.block_sizes[i];
            x1.blocks[i]      = create_block_region(bs);
            x1.block_sizes[i] = bs;
        }
    }
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif
