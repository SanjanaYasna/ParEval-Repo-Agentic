// Copyright 2013 Mario Mulansky
// Translated for Legion-oriented execution (runtime-agnostic local operation)
#ifndef SHARED_OPERATIONS_HPP
#define SHARED_OPERATIONS_HPP

#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

using dvec = std::vector<double>;
using shared_vec = std::shared_ptr<dvec>;

namespace legion_odeint_detail {

template <typename T>
struct is_shared_ptr : std::false_type {};

template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

// Access helper: supports both std::vector<double> and std::shared_ptr<std::vector<double>>.
template <typename V>
decltype(auto) as_vec(V& v)
{
    if constexpr (is_shared_ptr<std::decay_t<V>>::value) {
        return (*v);
    } else {
        return (v);
    }
}

template <typename V>
decltype(auto) as_vec(const V& v)
{
    if constexpr (is_shared_ptr<std::decay_t<V>>::value) {
        return (*v);
    } else {
        return (v);
    }
}

} // namespace legion_odeint_detail

struct local_legion_shared_operations
{
    template <typename Fac1, typename Fac2 = Fac1>
    struct scale_sum2
    {
        Fac1 m_alpha1{};
        Fac2 m_alpha2{};

        scale_sum2() = default;

        scale_sum2(Fac1 alpha1, Fac2 alpha2)
            : m_alpha1(alpha1), m_alpha2(alpha2)
        {}

        template <typename S1, typename S2, typename S3>
        S1 operator()(S1 x1, const S2& x2, const S3& x3) const
        {
            auto& out = legion_odeint_detail::as_vec(x1);
            const auto& in2 = legion_odeint_detail::as_vec(x2);
            const auto& in3 = legion_odeint_detail::as_vec(x3);

            const std::size_t n = out.size();
            for (std::size_t i = 0; i < n; ++i) {
                out[i] = m_alpha1 * in2[i] + m_alpha2 * in3[i];
            }
            return x1;
        }
    };
};

// Backward-compatible alias to minimize changes in other translated files.
using local_dataflow_shared_operations = local_legion_shared_operations;

#endif
