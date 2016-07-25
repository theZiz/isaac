/* This file is part of ISAAC.
 *
 * ISAAC is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * ISAAC is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with ISAAC.  If not, see <www.gnu.org/licenses/>. */

#pragma once

#include <boost/preprocessor.hpp>

#include "isaac_defines.hpp"

namespace isaac
{

typedef float isaac_float;
typedef int32_t isaac_int;
typedef uint32_t isaac_uint;

#define ISAAC_COMPONENTS_SEQ_3 (x)(y)(z)(w)
#define ISAAC_COMPONENTS_SEQ_2 (x)(y)(z)
#define ISAAC_COMPONENTS_SEQ_1 (x)(y)
#define ISAAC_COMPONENTS_SEQ_0 (x)

template <typename T,unsigned int d>
union Vector
{
	struct
	{
		T x;
		T y;
		T z;
		T w;
	} value;
	T direct[d];
};

//Specialization for 1…3 dimensional vectors with only (x), (x,y) or (x,y,z) in "vec".
#define ISAAC_SPECIALIZATION_DEF(Z, I, unused) \
	template <typename T> \
	union Vector<T, BOOST_PP_INC(I) > \
	{ \
		struct \
		{ \
			T BOOST_PP_SEQ_ENUM( BOOST_PP_CAT( ISAAC_COMPONENTS_SEQ_ , I ) ); \
		} value; \
		T direct[ BOOST_PP_INC(I) ]; \
	};
BOOST_PP_REPEAT(3, ISAAC_SPECIALIZATION_DEF, ~)
#undef ISAAC_SPECIALIZATION_DEF

template <typename T>
union Vector<T,0>
{
};

#ifdef __CUDACC__
	#define ISAAC_CUDA_DEF(Z, I, TYPE) \
		template <> \
		union Vector<TYPE,BOOST_PP_INC(I)> \
		{ \
			BOOST_PP_CAT(TYPE, BOOST_PP_INC(I) ) value; \
			TYPE direct[ BOOST_PP_INC(I) ]; \
		};
	BOOST_PP_REPEAT(4, ISAAC_CUDA_DEF, int)
	BOOST_PP_REPEAT(4, ISAAC_CUDA_DEF, float)
	BOOST_PP_REPEAT(4, ISAAC_CUDA_DEF, uint)
	#undef ISAAC_CUDA_DEF
#endif

template <typename T,unsigned int d,int c>
union VectorArray
{
	Vector<T,d> data[c];
	T array[d*c];
};

#define ISAAC_UNARY_OPERATOR_OVERLOAD( OPERATOR ) \
    /* Vector<T,d> OP Vector<T,d> */\
    template <typename T,unsigned int d> \
    const Vector<T,d> inline __host__ __device__ operator OPERATOR (Vector<T,d> const& lhs, Vector<T,d> const& rhs) \
    { \
        Vector<T,d> tmp(lhs); \
        for (unsigned i = 0; i < d; i++) \
            tmp.direct[i] = tmp.direct[i] OPERATOR rhs.direct[i]; \
        return tmp; \
    }; \
    \
    /* Vector<T,d> OP T */\
    template <typename T,unsigned int d> \
    const Vector<T,d> inline __host__ __device__ operator OPERATOR (Vector<T,d> const& lhs, T const& rhs) \
    { \
        Vector<T,d> tmp(lhs); \
        for (unsigned i = 0; i < d; i++) \
            tmp.direct[i] = tmp.direct[i] OPERATOR rhs; \
        return tmp; \
    }; \
    \
    /* T OP Vector<T,d> */\
    template <typename T,unsigned int d> \
    const Vector<T,d> inline __host__ __device__ operator OPERATOR (T const& lhs, Vector<T,d> const& rhs) \
    { \
        Vector<T,d> tmp(rhs); \
        for (unsigned i = 0; i < d; i++) \
            tmp.direct[i] = lhs OPERATOR tmp.direct[i]; \
        return tmp; \
    }; \
    \
    /* VectorArray<T,d,c> OP VectorArray<T,d,c> */\
    template <typename T,unsigned int d,int c> \
    const VectorArray<T,d,c> inline __host__ __device__ operator OPERATOR (VectorArray<T,d,c> const& lhs, VectorArray<T,d,c> const& rhs) \
    { \
        VectorArray<T,d,c> tmp(lhs); \
        for (unsigned i = 0; i < d*c; i++) \
            tmp.array[i] = tmp.array[i] OPERATOR rhs.array[i]; \
        return tmp; \
    }; \
    \
    /* VectorArray<T,d,c> OP T */\
    template <typename T,unsigned int d,int c> \
    const VectorArray<T,d,c> inline __host__ __device__ operator OPERATOR (VectorArray<T,d,c> const& lhs, T const& rhs) \
    { \
        VectorArray<T,d,c> tmp(lhs); \
        for (unsigned i = 0; i < d*c; i++) \
            tmp.array[i] = tmp.array[i] OPERATOR rhs; \
        return tmp; \
    }; \
    \
    /* T OP VectorArray<T,d,c> */\
    template <typename T,unsigned int d,int c> \
    const VectorArray<T,d,c> inline __host__ __device__ operator OPERATOR (T const& lhs, VectorArray<T,d,c> const& rhs) \
    { \
        VectorArray<T,d,c> tmp(rhs); \
        for (unsigned i = 0; i < d*c; i++) \
            tmp.array[i] = lhs OPERATOR tmp.array[i]; \
        return tmp; \
    };

ISAAC_UNARY_OPERATOR_OVERLOAD(+)
ISAAC_UNARY_OPERATOR_OVERLOAD(-)
ISAAC_UNARY_OPERATOR_OVERLOAD(*)
ISAAC_UNARY_OPERATOR_OVERLOAD(/)

#undef ISAAC_UNARY_OPERATOR_OVERLOAD


/* - Vector<T,d>*/\
template <typename T,unsigned int d>
const Vector<T,d> inline __host__ __device__ operator - (Vector<T,d> const& lhs)
{
    Vector<T,d> tmp(lhs);
    for (unsigned i = 0; i < d; i++)
        tmp.direct[i] = - tmp.direct[i];
    return tmp;
};

/* - VectorArray<T,d,c> */\
template <typename T,unsigned int d,int c>
const VectorArray<T,d,c> inline __host__ __device__ operator - (VectorArray<T,d,c> const& lhs)
{
    VectorArray<T,d,c> tmp(lhs);
    for (unsigned i = 0; i < d*c; i++)
        tmp.array[i] = - tmp.array[i];
    return tmp;
};

template < size_t simdim >
struct isaac_size_struct
{
    Vector<size_t,simdim> global_size;
    size_t max_global_size;
    Vector<size_t,simdim> position;
    Vector<size_t,simdim> local_size;
    Vector<size_t,simdim> global_size_scaled;
    size_t max_global_size_scaled;
    Vector<size_t,simdim> position_scaled;
    Vector<size_t,simdim> local_size_scaled;
};

template< int N >
struct transfer_d_struct
{
    Vector<float,4>* pointer[ N ];
};

template< int N >
struct transfer_h_struct
{
    Vector<float,4>* pointer[ N ];
    std::map< isaac_uint, Vector<float,4> > description[ N ];
};

struct functions_struct
{
    std::string source;
    isaac_int bytecode[ISAAC_MAX_FUNCTORS];
    isaac_int error_code;
};

template< int N >
struct source_weight_struct
{
    isaac_float value[ N ];
};

template< int N >
struct pointer_array_struct
{
    void* pointer[ N ];
};

struct minmax_struct
{
    isaac_float min;
    isaac_float max;
};

template< int N>
struct minmax_array_struct
{
    isaac_float min[ N ];
    isaac_float max[ N ];
};

struct clipping_struct
{
    ISAAC_HOST_DEVICE_INLINE clipping_struct() :
        count(0)
    {}
    isaac_uint count;
    struct
    {
        Vector<float,3> position;
        Vector<float,3> normal;
    } elem[ ISAAC_MAX_CLIPPING ];
};



typedef enum
{
    META_MERGE = 0,
    META_MASTER = 1
} IsaacVisualizationMetaEnum;

} //namespace isaac;
