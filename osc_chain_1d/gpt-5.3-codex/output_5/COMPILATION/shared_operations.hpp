// Translated from HPX-oriented helper to a Legion-friendly (runtime-agnostic) helper.
// This file contains only local arithmetic operations and can be used inside Legion tasks.
#ifndef SHARED_OPERATIONS_HPP
#define SHARED_OPERATIONS_HPP

#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>

typedef std::vector<double> dvec;
typedef std::shared_ptr<dvec> shared_vec;

struct local_dataflow_shared_operations
{
    template <typename Fac1, typename Fac2 = Fac1>
    struct scale_sum2
    {
        const Fac1 m_alpha1;
        const Fac2 m_alpha2;

        scale_sum2() : m_alpha1(0), m_alpha2(0) {}
        scale_sum2(Fac1 alpha1, Fac2 alpha2)
            : m_alpha1(alpha1), m_alpha2(alpha2) {}

        template <typename S1, typename S2, typename S3>
        S1 operator()(S1 x1, const S2 x2, const S3 x3) const
        {
            const std::size_t n = size_of(x1);
            assert(n == size_of(x2) && n == size_of(x3));

            for (std::size_t i = 0; i < n; ++i) {
                elem(x1, i) = m_alpha1 * elem(x2, i) + m_alpha2 * elem(x3, i);
            }
            return x1;
        }

    private:
        // Generic container support
        template <typename V>
        static std::size_t size_of(const V& v) { return v.size(); }

        template <typename V>
        static double& elem(V& v, std::size_t i) { return v[i]; }

        template <typename V>
        static const double& elem(const V& v, std::size_t i) { return v[i]; }

        // shared_ptr container support
        template <typename V>
        static std::size_t size_of(const std::shared_ptr<V>& v) { return v->size(); }

        template <typename V>
        static double& elem(std::shared_ptr<V>& v, std::size_t i) { return (*v)[i]; }

        template <typename V>
        static const double& elem(const std::shared_ptr<V>& v, std::size_t i) { return (*v)[i]; }
    };
};

#endif
