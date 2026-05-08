// Copyright 2013 Mario Mulansky
// Resizing functionality for odeint (Legion version)
#ifndef SHARED_RESIZE_HPP
#define SHARED_RESIZE_HPP

#include <memory>
#include <vector>

#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/util/resize.hpp>
#include <boost/numeric/odeint/util/same_size.hpp>
#include <boost/numeric/odeint/util/state_wrapper.hpp>

// Legion runtime is used in the rest of the codebase, but this header only
// needs host-side state container definitions.
#include <legion.h>

typedef std::vector<double> dvec;
typedef std::shared_ptr<dvec> shared_vec;

// In the Legion translation we keep concrete block storage directly.
// (HPX shared_future<shared_vec> -> shared_vec)
typedef std::vector<shared_vec> state_type;

namespace boost {
namespace numeric {
namespace odeint {

template<>
struct is_resizeable<state_type>
{
    typedef boost::true_type type;
    static const bool value = type::value;
};

template<>
struct same_size_impl<state_type, state_type>
{
    static bool same_size(const state_type &x1, const state_type &x2)
    {
        return x1.size() == x2.size();
    }
};

template<>
struct resize_impl<state_type, state_type>
{
    static void resize(state_type &x1, const state_type &x2)
    {
        x1.resize(x2.size());
        for (size_t i = 0; i < x2.size(); ++i)
        {
            const size_t n = (x2[i] ? x2[i]->size() : 0);
            x1[i] = std::make_shared<dvec>(n);
        }
    }
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif // SHARED_RESIZE_HPP
