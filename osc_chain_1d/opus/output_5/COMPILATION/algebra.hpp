// Copyright 2013 Mario Mulansky
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include <cstddef>

struct local_dataflow_algebra
{
    template< typename S , typename Op >
    void for_each3( S &s1 , const S &s2 , const S &s3 , Op op )
    {
        const size_t N = s1.size();
        for( size_t i = 0 ; i < N ; ++i )
        {
            s1[i] = op( s1[i] , s2[i] , s3[i] );
        }
    }
};

#endif
