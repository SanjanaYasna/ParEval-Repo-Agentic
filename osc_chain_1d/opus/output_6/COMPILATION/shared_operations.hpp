// Copyright 2013 Mario Mulansky
// Adapted for Legion execution model
#ifndef SHARED_OPERATIONS_HPP
#define SHARED_OPERATIONS_HPP

#include "legion.h"

using namespace Legion;

// Field ID for the scalar double values stored in logical regions
enum FieldIDs {
    FID_VAL = 101,
};

struct local_dataflow_shared_operations
{
    template< typename Fac1 , typename Fac2=Fac1 >
    struct scale_sum2
    {
        Fac1 m_alpha1;
        Fac2 m_alpha2;

        scale_sum2()
            : m_alpha1( 0 ) , m_alpha2( 0 )
        {}

        scale_sum2( Fac1 alpha1 , Fac2 alpha2 )
            : m_alpha1( alpha1 ) , m_alpha2( alpha2 )
        { }

        // Apply the scale_sum2 operation using Legion field accessors
        // Computes: x1[i] = alpha1 * x2[i] + alpha2 * x3[i]  for all i in rect
        void operator() ( const FieldAccessor<READ_WRITE, double, 1> &x1 ,
                          const FieldAccessor<READ_ONLY, double, 1> &x2 ,
                          const FieldAccessor<READ_ONLY, double, 1> &x3 ,
                          const Rect<1> &rect ) const
        {
            for( PointInRectIterator<1> pir(rect); pir(); pir++ )
                x1[*pir] = m_alpha1 * x2[*pir] + m_alpha2 * x3[*pir];
        }
    };
};

#endif
