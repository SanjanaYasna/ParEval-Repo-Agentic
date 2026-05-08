// Copyright 2013 Mario Mulansky
// Adapted for the Legion execution model
#ifndef SHARED_OPERATIONS_HPP
#define SHARED_OPERATIONS_HPP

#include <cstddef>
#include "legion.h"

using namespace Legion;

// Field ID used for the double values stored in logical regions
enum FieldIDs {
    FID_VAL = 101,
};

// Serializable argument struct for passing scale_sum2 coefficients
// as TaskArgument when launching Legion tasks
struct ScaleSum2Args {
    double alpha1;
    double alpha2;
};

struct local_dataflow_shared_operations
{
    template< typename Fac1 , typename Fac2 = Fac1 >
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

        // Apply on raw double arrays — called from within Legion task bodies
        // after extracting pointers from FieldAccessors.
        // Computes: x1[i] = alpha1 * x2[i] + alpha2 * x3[i]
        void operator() ( double *x1 , const double *x2 , const double *x3 , size_t n ) const
        {
            for( size_t i = 0 ; i < n ; ++i )
                x1[i] = m_alpha1 * x2[i] + m_alpha2 * x3[i];
        }

        // Convenience overload that works directly with Legion FieldAccessors
        // over a 1-D domain (e.g. a sub-region obtained from a partition).
        template< typename ACC_WR , typename ACC_RD1 , typename ACC_RD2 >
        void apply( ACC_WR acc_x1 ,
                    ACC_RD1 acc_x2 ,
                    ACC_RD2 acc_x3 ,
                    const Domain &dom ) const
        {
            for( Domain::DomainPointIterator itr(dom) ; itr ; itr++ )
            {
                acc_x1[*itr] = m_alpha1 * (double)acc_x2[*itr]
                             + m_alpha2 * (double)acc_x3[*itr];
            }
        }
    };
};

#endif
