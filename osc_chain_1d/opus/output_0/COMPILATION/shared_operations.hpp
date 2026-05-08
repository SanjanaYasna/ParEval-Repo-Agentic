// Copyright 2013 Mario Mulansky
#ifndef SHARED_OPERATIONS_HPP
#define SHARED_OPERATIONS_HPP

#include <cstddef>
#include "legion.h"

using namespace Legion;

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

        // Operates on raw double arrays extracted from Legion region accessors.
        // x1 is the output buffer, x2 and x3 are input buffers, N is element count.
        void operator() ( double *x1 , const double *x2 , const double *x3 , size_t N ) const
        {
            for( size_t i=0 ; i<N ; ++i )
                x1[i] = m_alpha1*x2[i] + m_alpha2*x3[i];
        }
    };
};

#endif
