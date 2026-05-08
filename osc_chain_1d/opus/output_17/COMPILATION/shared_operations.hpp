// Copyright 2013 Mario Mulansky
#ifndef SHARED_OPERATIONS_HPP
#define SHARED_OPERATIONS_HPP

#include <cstddef>
#include <legion.h>

using namespace Legion;

// Field ID for storing double values in state logical regions
enum FieldIDs {
    FID_VAL = 0,
};

struct local_dataflow_shared_operations
{
    template< typename Fac1 , typename Fac2=Fac1 >
    struct scale_sum2
    {
        const Fac1 m_alpha1;
        const Fac2 m_alpha2;

        scale_sum2()
            : m_alpha1( 0 ) , m_alpha2( 0 )
        {}

        scale_sum2( Fac1 alpha1 , Fac2 alpha2 )
            : m_alpha1( alpha1 ) , m_alpha2( alpha2 )
        { }

        // Apply the scale_sum2 operation using Legion FieldAccessors.
        // Computes: x1[i] = alpha1 * x2[i] + alpha2 * x3[i]
        // Called from within a Legion task body by the algebra.
        void operator() ( const FieldAccessor<READ_WRITE, double, 1> &x1 ,
                          const FieldAccessor<READ_ONLY,  double, 1> &x2 ,
                          const FieldAccessor<READ_ONLY,  double, 1> &x3 ,
                          const Rect<1> &bounds ) const
        {
            for( PointInRectIterator<1> pir(bounds); pir(); pir++ )
            {
                x1[*pir] = m_alpha1 * x2[*pir] + m_alpha2 * x3[*pir];
            }
        }

        // Apply the operation on raw double pointers (e.g. for local computation)
        void operator() ( double *x1 , const double *x2 , const double *x3 ,
                          size_t N ) const
        {
            for( size_t i=0 ; i<N ; ++i )
                x1[i] = m_alpha1 * x2[i] + m_alpha2 * x3[i];
        }
    };
};

#endif
