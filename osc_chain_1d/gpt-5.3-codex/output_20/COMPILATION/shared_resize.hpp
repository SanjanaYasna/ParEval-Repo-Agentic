// Copyright 2013 Mario Mulansky
// resizing functionality for odeint (Legion translation)
#ifndef SHARED_RESIZE_HPP
#define SHARED_RESIZE_HPP

#include <memory>
#include <vector>

#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/util/resize.hpp>
#include <boost/numeric/odeint/util/same_size.hpp>

typedef std::vector<double> dvec;
typedef std::shared_ptr<dvec> shared_vec;

// Legion version: keep block-based state, but without HPX futures/dataflow.
typedef std::vector<shared_vec> state_type;

namespace boost {
namespace numeric {
namespace odeint {

template <>
struct is_resizeable<state_type>
{
    typedef boost::true_type type;
    static const bool value = type::value;
};

template <>
struct same_size_impl<state_type, state_type>
{
    static bool same_size(const state_type& x1, const state_type& x2)
    {
        return x1.size() == x2.size();
    }
};

template <>
struct resize_impl<state_type, state_type>
{
    static void resize(state_type& x1, const state_type& x2)
    {
        x1.resize(x2.size());
        for (size_t i = 0; i < x2.size(); ++i)
        {
            if (x2[i])
                x1[i] = std::make_shared<dvec>(x2[i]->size());
            else
                x1[i] = std::make_shared<dvec>();
        }
    }
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif // SHARED_RESIZE_HPP
