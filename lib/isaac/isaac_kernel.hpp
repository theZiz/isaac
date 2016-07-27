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

#include "isaac_macros.hpp"
#include "isaac_fusion_extension.hpp"
#include "isaac_functors.hpp"

#include <boost/mpl/at.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/back.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/fusion/include/push_back.hpp>
#include <boost/mpl/size.hpp>

#include <float.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wsign-compare"

namespace isaac
{

namespace fus = boost::fusion;
namespace mpl = boost::mpl;

typedef isaac_float (*isaac_functor_chain_pointer_4)(Vector<float,4>, isaac_int );
typedef isaac_float (*isaac_functor_chain_pointer_3)(Vector<float,3>, isaac_int );
typedef isaac_float (*isaac_functor_chain_pointer_2)(Vector<float,2>, isaac_int );
typedef isaac_float (*isaac_functor_chain_pointer_1)(Vector<float,1>, isaac_int );
typedef isaac_float (*isaac_functor_chain_pointer_N)(void*              , isaac_int );

ISAAC_CONSTANT isaac_float isaac_inverse_d[16];
ISAAC_CONSTANT isaac_size_struct<3> isaac_size_d[1]; //[1] to access it for cuda and alpaka the same way
ISAAC_CONSTANT Vector<float,4> isaac_parameter_d[ ISAAC_MAX_SOURCES*ISAAC_MAX_FUNCTORS ];
ISAAC_CONSTANT isaac_functor_chain_pointer_N isaac_function_chain_d[ ISAAC_MAX_SOURCES ];


template
<
    typename TFunctorVector,
    int TFeatureDim,
    int NR
>
struct FillFunctorChainPointerKernelStruct
{
    ISAAC_DEVICE static isaac_functor_chain_pointer_N call( isaac_int const * const bytecode )
    {
        #define ISAAC_SUB_CALL(Z, I, U) \
            if (bytecode[ISAAC_MAX_FUNCTORS-NR] == I) \
                return FillFunctorChainPointerKernelStruct \
                < \
                    typename mpl::push_back< TFunctorVector, typename boost::mpl::at_c<IsaacFunctorPool,I>::type >::type, \
                    TFeatureDim, \
                    NR - 1 \
                > ::call( bytecode );
        BOOST_PP_REPEAT( ISAAC_FUNCTOR_COUNT, ISAAC_SUB_CALL, ~)
        #undef ISAAC_SUB_CALL
        return NULL; //Should never be reached anyway
    }
};

template
<
    typename TFunctorVector,
    int TFeatureDim
>
ISAAC_DEVICE isaac_float applyFunctorChain (
    Vector<float, TFeatureDim > const value,
    isaac_int const src_id
)
{
    #define  ISAAC_LEFT_DEF(Z,I,U) mpl::at_c< TFunctorVector, ISAAC_MAX_FUNCTORS - I - 1 >::type::call(
    #define ISAAC_RIGHT_DEF(Z,I,U) , isaac_parameter_d[ src_id * ISAAC_MAX_FUNCTORS + I ] )
    #define  ISAAC_LEFT BOOST_PP_REPEAT( ISAAC_MAX_FUNCTORS, ISAAC_LEFT_DEF, ~)
    #define ISAAC_RIGHT BOOST_PP_REPEAT( ISAAC_MAX_FUNCTORS, ISAAC_RIGHT_DEF, ~)
    // expands to: funcN( ... func1( func0( data, p[0] ), p[1] ) ... p[N] );
    return ISAAC_LEFT value ISAAC_RIGHT .value.x;
    #undef ISAAC_LEFT_DEF
    #undef ISAAC_LEFT
    #undef ISAAC_RIGHT_DEF
    #undef ISAAC_RIGHT
}


template
<
    typename TFunctorVector,
    int TFeatureDim
>
struct FillFunctorChainPointerKernelStruct
<
    TFunctorVector,
    TFeatureDim,
    0 //<- Specialization
>
{
    ISAAC_DEVICE static isaac_functor_chain_pointer_N call( isaac_int const * const bytecode)
    {
        return reinterpret_cast<isaac_functor_chain_pointer_N>(applyFunctorChain<TFunctorVector,TFeatureDim>);
    }
};


#if ISAAC_ALPAKA == 1
    struct fillFunctorChainPointerKernel
    {
        template <typename TAcc__>
        ALPAKA_FN_ACC void operator()(
            TAcc__ const &acc,
#else
        __global__ void fillFunctorChainPointerKernel(
#endif
            isaac_functor_chain_pointer_N * const functor_chain_d)
#if ISAAC_ALPAKA == 1
        const
#endif
        {
            isaac_int bytecode[ISAAC_MAX_FUNCTORS];
            for (int i = 0; i < ISAAC_MAX_FUNCTORS; i++)
                bytecode[i] = 0;
            for (int i = 0; i < ISAAC_FUNCTOR_COMPLEX; i++)
            {
                functor_chain_d[i*4+0] = FillFunctorChainPointerKernelStruct<mpl::vector<>,1,ISAAC_MAX_FUNCTORS>::call( bytecode );
                functor_chain_d[i*4+1] = FillFunctorChainPointerKernelStruct<mpl::vector<>,2,ISAAC_MAX_FUNCTORS>::call( bytecode );
                functor_chain_d[i*4+2] = FillFunctorChainPointerKernelStruct<mpl::vector<>,3,ISAAC_MAX_FUNCTORS>::call( bytecode );
                functor_chain_d[i*4+3] = FillFunctorChainPointerKernelStruct<mpl::vector<>,4,ISAAC_MAX_FUNCTORS>::call( bytecode );
                for (int j = ISAAC_MAX_FUNCTORS - 1; j >= 0; j--)
                    if ( bytecode[j] < ISAAC_FUNCTOR_COUNT-1 )
                    {
                        bytecode[j]++;
                        break;
                    }
                    else
                        bytecode[j] = 0;
            }
        }
#if ISAAC_ALPAKA == 1
    };
#endif

template <
    isaac_int TInterpolation,
    typename NR,
    unsigned int isaac_vector_elem,
    typename TSource,
    typename TPos,
    typename TPointerArray,
    typename TLocalSize,
    typename TScale,
    typename TLoopFinish
>
ISAAC_HOST_DEVICE_INLINE VectorArray<isaac_float,1,isaac_vector_elem> get_value (
    const TSource& source,
    const TPos& pos, //array
    const TPointerArray& pointerArray,
    const TLocalSize& local_size,
    const TScale& scale,
    const TLoopFinish& loop_finish //array
)
{
    VectorArray<isaac_float, TSource::feature_dim, isaac_vector_elem > data;
    Vector<isaac_float, TSource::feature_dim >* ptr = (Vector<isaac_float, TSource::feature_dim >*)(pointerArray.pointer[ NR::value ] );
    VectorArray<isaac_float,1,isaac_vector_elem> result;
    VectorArray<isaac_int,3,isaac_vector_elem> coord;
    if (TInterpolation == 0)
    {
        ISAAC_ELEM_ITERATE_LEN(e,isaac_vector_elem)
            coord.data[e].value =
            {
                isaac_int(pos.data[e].value.x),
                isaac_int(pos.data[e].value.y),
                isaac_int(pos.data[e].value.z)
            };
        if (TSource::persistent)
        {
            ISAAC_ELEM_ITERATE_LEN(e,isaac_vector_elem)
                if ( ISAAC_CHECK_FINISH(loop_finish) )
                    data.data[e] = source[coord.data[e]];
        }
        else
            ISAAC_ELEM_ITERATE_LEN(e,isaac_vector_elem)
                if ( ISAAC_CHECK_FINISH(loop_finish) )
                    data.data[e] = ptr[coord.data[e].value.x + ISAAC_GUARD_SIZE + (coord.data[e].value.y + ISAAC_GUARD_SIZE) * (local_size.value.x + 2 * ISAAC_GUARD_SIZE) + (coord.data[e].value.z + ISAAC_GUARD_SIZE) * ( (local_size.value.x + 2 * ISAAC_GUARD_SIZE) * (local_size.value.y + 2 * ISAAC_GUARD_SIZE) )];
    }
    else
    {
        VectorArray<float, TSource::feature_dim, isaac_vector_elem > data8[2][2][2];
        for (int x = 0; x < 2; x++)
            for (int y = 0; y < 2; y++)
                for (int z = 0; z < 2; z++)
                {
                    ISAAC_ELEM_ITERATE_LEN(e,isaac_vector_elem)
                        coord.data[e].value.x = isaac_int(x?ceil(pos.data[e].value.x):floor(pos.data[e].value.x));
                    ISAAC_ELEM_ITERATE_LEN(e,isaac_vector_elem)
                        coord.data[e].value.y = isaac_int(y?ceil(pos.data[e].value.y):floor(pos.data[e].value.y));
                    ISAAC_ELEM_ITERATE_LEN(e,isaac_vector_elem)
                        coord.data[e].value.z = isaac_int(z?ceil(pos.data[e].value.z):floor(pos.data[e].value.z));
                    if (!TSource::has_guard && TSource::persistent)
                    {
                        ISAAC_ELEM_ITERATE_LEN(e,isaac_vector_elem)
                            if ( isaac_uint(coord.data[e].value.x) >= local_size.value.x )
                                coord.data[e].value.x = isaac_int(x?floor(pos.data[e].value.x):ceil(pos.data[e].value.x));
                        ISAAC_ELEM_ITERATE_LEN(e,isaac_vector_elem)
                            if ( isaac_uint(coord.data[e].value.y) >= local_size.value.y )
                                coord.data[e].value.y = isaac_int(y?floor(pos.data[e].value.y):ceil(pos.data[e].value.y));
                        ISAAC_ELEM_ITERATE_LEN(e,isaac_vector_elem)
                            if ( isaac_uint(coord.data[e].value.z) >= local_size.value.z )
                                coord.data[e].value.z = isaac_int(z?floor(pos.data[e].value.z):ceil(pos.data[e].value.z));
                    }
                    if (TSource::persistent)
                    {
                        ISAAC_ELEM_ITERATE_LEN(e,isaac_vector_elem)
                            if ( ISAAC_CHECK_FINISH(loop_finish) )
                                data8[x][y][z].data[e] = source[coord.data[e]];
                    }
                    else
                        ISAAC_ELEM_ITERATE_LEN(e,isaac_vector_elem)
                            if ( ISAAC_CHECK_FINISH(loop_finish) )
                                data8[x][y][z].data[e] = ptr[coord.data[e].value.x + ISAAC_GUARD_SIZE + (coord.data[e].value.y + ISAAC_GUARD_SIZE) * (local_size.value.x + 2 * ISAAC_GUARD_SIZE) + (coord.data[e].value.z + ISAAC_GUARD_SIZE) * ( (local_size.value.x + 2 * ISAAC_GUARD_SIZE) * (local_size.value.y + 2 * ISAAC_GUARD_SIZE) )];
                }
        VectorArray<float, 3, isaac_vector_elem > pos_in_cube;
        ISAAC_ELEM_ITERATE_LEN(e,isaac_vector_elem)
            pos_in_cube.data[e].value =
            {
                pos.data[e].value.x - floor(pos.data[e].value.x),
                pos.data[e].value.y - floor(pos.data[e].value.y),
                pos.data[e].value.z - floor(pos.data[e].value.z)
            };
        VectorArray<float, TSource::feature_dim, isaac_vector_elem > data4[2][2];
        for (int x = 0; x < 2; x++)
            for (int y = 0; y < 2; y++)
                ISAAC_ELEM_ITERATE_LEN(e,isaac_vector_elem)
                    data4[x][y].data[e] =
                        data8[x][y][0].data[e] * (isaac_float(1) - pos_in_cube.data[e].value.z) +
                        data8[x][y][1].data[e] * (                 pos_in_cube.data[e].value.z);
        VectorArray<float, TSource::feature_dim,isaac_vector_elem > data2[2];
        for (int x = 0; x < 2; x++)
            ISAAC_ELEM_ITERATE_LEN(e,isaac_vector_elem)
                data2[x].data[e] =
                    data4[x][0].data[e] * (isaac_float(1) - pos_in_cube.data[e].value.y) +
                    data4[x][1].data[e] * (                 pos_in_cube.data[e].value.y);
        ISAAC_ELEM_ITERATE_LEN(e,isaac_vector_elem)
            data.data[e] =
                data2[0].data[e] * (isaac_float(1) - pos_in_cube.data[e].value.x) +
                data2[1].data[e] * (                 pos_in_cube.data[e].value.x);
    }
    result = Vector<isaac_float,1>({0});

    #if ISAAC_ALPAKA == 1 || defined(__CUDA_ARCH__)
    if (TSource::feature_dim == 1)
            ISAAC_ELEM_ITERATE_LEN(e,isaac_vector_elem)
                result.array[e] = reinterpret_cast<isaac_functor_chain_pointer_1>(isaac_function_chain_d[ NR::value ])( *(reinterpret_cast< Vector<float,1>* >(&(data.data[e]))), NR::value );
        if (TSource::feature_dim == 2)
            ISAAC_ELEM_ITERATE_LEN(e,isaac_vector_elem)
                result.array[e] = reinterpret_cast<isaac_functor_chain_pointer_2>(isaac_function_chain_d[ NR::value ])( *(reinterpret_cast< Vector<float,2>* >(&(data.data[e]))), NR::value );
        if (TSource::feature_dim == 3)
            ISAAC_ELEM_ITERATE_LEN(e,isaac_vector_elem)
                result.array[e] = reinterpret_cast<isaac_functor_chain_pointer_3>(isaac_function_chain_d[ NR::value ])( *(reinterpret_cast< Vector<float,3>* >(&(data.data[e]))), NR::value );
        if (TSource::feature_dim == 4)
            ISAAC_ELEM_ITERATE_LEN(e,isaac_vector_elem)
                result.array[e] = reinterpret_cast<isaac_functor_chain_pointer_4>(isaac_function_chain_d[ NR::value ])( *(reinterpret_cast< Vector<float,4>* >(&(data.data[e]))), NR::value );
    #endif
    return result;
}

template < typename TLocalSize >
ISAAC_HOST_DEVICE_INLINE void check_coord( Vector<float,3>& coord, const TLocalSize local_size)
{
    if (coord.value.x < isaac_float(0))
        coord.value.x = isaac_float(0);
    if (coord.value.y < isaac_float(0))
        coord.value.y = isaac_float(0);
    if (coord.value.z < isaac_float(0))
        coord.value.z = isaac_float(0);
    if ( coord.value.x >= isaac_float(local_size.value.x) )
        coord.value.x = isaac_float(local_size.value.x)-isaac_float(1);
    if ( coord.value.y >= isaac_float(local_size.value.y) )
        coord.value.y = isaac_float(local_size.value.y)-isaac_float(1);
    if ( coord.value.z >= isaac_float(local_size.value.z) )
        coord.value.z = isaac_float(local_size.value.z)-isaac_float(1);
}

template <
    size_t Ttransfer_size,
    typename TFilter,
    isaac_int TInterpolation,
    isaac_int TIsoSurface
>
struct merge_source_iterator
{
    template
    <
        typename NR,
        typename TSource,
        typename TColor,
        typename TPos,
        typename TLocalSize,
        typename TTransferArray,
        typename TSourceWeight,
        typename TPointerArray,
        typename TStep,
        typename TStepLength,
        typename TScale,
        typename TLoopFinish
    >
    ISAAC_HOST_DEVICE_INLINE void operator()(
        const NR& nr,
        const TSource& source,
        TColor& color, //array
        const TPos& pos, //array
        const TLocalSize& local_size,
        const TTransferArray& transferArray,
        const TSourceWeight& sourceWeight,
        const TPointerArray& pointerArray,
        const TStep& step, //array
        const TStepLength& stepLength,
        const TScale& scale,
        TLoopFinish& loop_finish //array
    ) const
    {
        if ( mpl::at_c< TFilter, NR::value >::type::value )
        {
            VectorArray<isaac_float,1,ISAAC_VECTOR_ELEM> result;
            VectorArray<isaac_int,1,ISAAC_VECTOR_ELEM> lookup_value;
            VectorArray<isaac_float,4,ISAAC_VECTOR_ELEM> value;
            result = get_value< TInterpolation, NR, ISAAC_VECTOR_ELEM >( source, pos, pointerArray, local_size, scale, loop_finish );
            ISAAC_ELEM_ITERATE(e)
                lookup_value.array[e] = isaac_int( round(result.array[e] * isaac_float( Ttransfer_size ) ) );
            ISAAC_ELEM_ITERATE(e)
                if (lookup_value.array[e] < 0 )
                    lookup_value.array[e] = 0;
            ISAAC_ELEM_ITERATE(e)
                if (lookup_value.array[e] >= Ttransfer_size )
                    lookup_value.array[e] = Ttransfer_size - 1;
            ISAAC_ELEM_ITERATE(e)
                value.data[e] = transferArray.pointer[ NR::value ][ lookup_value.array[e] ];
            if (TIsoSurface)
            {
                ISAAC_ELEM_ITERATE(e)
                    if (value.data[e].value.w >= isaac_float(0.5) && ISAAC_CHECK_FINISH(loop_finish))
                    {
                        VectorArray<float,3,1>  left = {-1, 0, 0};
                        left.data[0] = left.data[0] + pos.data[e];
                        if (!TSource::has_guard && TSource::persistent)
                            check_coord( left.data[0], local_size);
                        VectorArray<float,3,1> right = { 1, 0, 0};
                        right.data[0] = right.data[0] + pos.data[e];
                        if (!TSource::has_guard && TSource::persistent)
                            check_coord( right.data[0], local_size );
                        isaac_float d1;
                        if (TInterpolation)
                            d1 = right.data[0].value.x - left.data[0].value.x;
                        else
                            d1 = isaac_int(right.data[0].value.x) - isaac_int(left.data[0].value.x);

                        VectorArray<float,3,1>    up = { 0,-1, 0};
                        up.data[0] = up.data[0] + pos.data[e];
                        if (!TSource::has_guard && TSource::persistent)
                            check_coord( up.data[0], local_size );
                        VectorArray<float,3,1>  down = { 0, 1, 0};
                        down.data[0] = down.data[0] + pos.data[e];
                        if (!TSource::has_guard && TSource::persistent)
                            check_coord( down.data[0], local_size );
                        isaac_float d2;
                        if (TInterpolation)
                            d2 = down.data[0].value.y - up.data[0].value.y;
                        else
                            d2 = isaac_int(down.data[0].value.y) - isaac_int(up.data[0].value.y);

                        VectorArray<float,3,1> front = { 0, 0,-1};
                        front.data[0] = front.data[0] + pos.data[e];
                        if (!TSource::has_guard && TSource::persistent)
                            check_coord( front.data[0], local_size );
                        VectorArray<float,3,1>  back = { 0, 0, 1};
                        back.data[0] = back.data[0] + pos.data[e];
                        if (!TSource::has_guard && TSource::persistent)
                            check_coord( back.data[0], local_size );
                        isaac_float d3;
                        if (TInterpolation)
                            d3 = back.data[0].value.z - front.data[0].value.z;
                        else
                            d3 = isaac_int(back.data[0].value.z) - isaac_int(front.data[0].value.z);

                        Vector<float,3> gradient=
                        {
                            (get_value< TInterpolation, NR, 1 >( source, right, pointerArray, local_size, scale, VectorArray<bool,1,1>({0}) ).array[0] -
                             get_value< TInterpolation, NR, 1 >( source,  left, pointerArray, local_size, scale, VectorArray<bool,1,1>({0}) ).array[0]) / d1,
                            (get_value< TInterpolation, NR, 1 >( source,  down, pointerArray, local_size, scale, VectorArray<bool,1,1>({0}) ).array[0] -
                             get_value< TInterpolation, NR, 1 >( source,    up, pointerArray, local_size, scale, VectorArray<bool,1,1>({0}) ).array[0]) / d2,
                            (get_value< TInterpolation, NR, 1 >( source,  back, pointerArray, local_size, scale, VectorArray<bool,1,1>({0}) ).array[0] -
                             get_value< TInterpolation, NR, 1 >( source, front, pointerArray, local_size, scale, VectorArray<bool,1,1>({0}) ).array[0]) / d3
                        };
                        isaac_float l = sqrt(
                            gradient.value.x * gradient.value.x +
                            gradient.value.y * gradient.value.y +
                            gradient.value.z * gradient.value.z
                        );
                        if (l == isaac_float(0))
                            color.data[e] = value.data[e];
                        else
                        {
                            gradient = gradient / l;
                            Vector<float,3> light = step.data[e] / stepLength;
                            isaac_float ac = fabs(
                                gradient.value.x * light.value.x +
                                gradient.value.y * light.value.y +
                                gradient.value.z * light.value.z );
                            #if ISAAC_SPECULAR == 1
                                color.data[e].value.x = value.data[e].value.x * ac + ac * ac * ac * ac;
                                color.data[e].value.y = value.data[e].value.y * ac + ac * ac * ac * ac;
                                color.data[e].value.z = value.data[e].value.z * ac + ac * ac * ac * ac;
                            #else
                                color.data[e].value.x = value.data[e].value.x * ac;
                                color.data[e].value.y = value.data[e].value.y * ac;
                                color.data[e].value.z = value.data[e].value.z * ac;
                            #endif
                        }
                        color.data[e].value.w = isaac_float(-1);
                        loop_finish.array[e] = true;
                    }
            }
            else
            {
                ISAAC_ELEM_ITERATE(e)
                    value.data[e].value.w *= sourceWeight.value[ NR::value ];
                ISAAC_ELEM_ITERATE(e)
                    value.data[e].value.x *= value.data[e].value.w;
                ISAAC_ELEM_ITERATE(e)
                    value.data[e].value.y *= value.data[e].value.w;
                ISAAC_ELEM_ITERATE(e)
                    value.data[e].value.z *= value.data[e].value.w;
                color = color + value;
            }
        }
    }
};

template <
    typename TFilter
>
struct check_no_source_iterator
{
    template
    <
        typename NR,
        typename TSource,
        typename TResult
    >
    ISAAC_HOST_DEVICE_INLINE  void operator()(
        const NR& nr,
        const TSource& source,
        TResult& result
    ) const
    {
        result |= mpl::at_c< TFilter, NR::value >::type::value;
    }
};


template <
    typename TSimDim,
    typename TSourceList,
    typename TTransferArray,
    typename TSourceWeight,
    typename TPointerArray,
    typename TFilter,
    size_t Ttransfer_size,
    isaac_int TInterpolation,
    isaac_int TIsoSurface,
    typename TScale
>
#if ISAAC_ALPAKA == 1
    struct isaacRenderKernel
    {
        template <typename TAcc__>
        ALPAKA_FN_ACC void operator()(
            TAcc__ const &acc,
#else
        __global__ void isaacRenderKernel(
#endif
            uint32_t * const pixels,
            const Vector<size_t,2> framebuffer_size,
            const Vector<uint32_t,2> framebuffer_start,
            const TSourceList sources,
            isaac_float step,
            const Vector<float,4> background_color,
            const TTransferArray transferArray,
            const TSourceWeight sourceWeight,
            const TPointerArray pointerArray,
            const TScale scale,
            const clipping_struct input_clipping)
#if ISAAC_ALPAKA == 1
        const
#endif
        {
            VectorArray<uint32_t,2,ISAAC_VECTOR_ELEM> pixel;
            VectorArray<bool,1,ISAAC_VECTOR_ELEM> finish;
#if ISAAC_ALPAKA == 1
            auto threadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            ISAAC_ELEM_ITERATE(e)
            {
                pixel.data[e].value.x = isaac_uint(threadIdx[2]) * isaac_uint(ISAAC_VECTOR_ELEM) + e;
                pixel.data[e].value.y = isaac_uint(threadIdx[1]);
#else
            ISAAC_ELEM_ITERATE(e)
            {
                pixel.data[e].value.x = isaac_uint(threadIdx.x + blockIdx.x * blockDim.x) * isaac_uint(ISAAC_VECTOR_ELEM) + e;
                pixel.data[e].value.y = isaac_uint(threadIdx.y + blockIdx.y * blockDim.y);
#endif
                finish.array[e] = false;
                pixel.data[e] = pixel.data[e] + framebuffer_start;
                if ( ISAAC_FOR_EACH_DIM_TWICE(2, pixel.data[e], >= framebuffer_size, || ) 0 )
                    finish.array[e] = true;
            }
            ISAAC_ELEM_ALL_TRUE_RETURN( finish.array )

            VectorArray<bool,1,ISAAC_VECTOR_ELEM> at_least_one;
            VectorArray<float,4,ISAAC_VECTOR_ELEM> color;

            ISAAC_ELEM_ITERATE(e)
            {
                color.data[e] = background_color;
                at_least_one.array[e] = true;
                isaac_for_each_with_mpl_params( sources, check_no_source_iterator<TFilter>(), at_least_one.array[e] );
                if (!at_least_one.array[e])
                {
                    if ( ISAAC_CHECK_FINISH(finish) )
                        ISAAC_SET_COLOR( pixels[pixel.data[e].value.x + pixel.data[e].value.y * framebuffer_size.value.x], color.data[e] )
                    finish.array[e] = true;
                }
            }
            ISAAC_ELEM_ALL_TRUE_RETURN( finish.array )

            VectorArray<float,2,ISAAC_VECTOR_ELEM> pixel_f;
            VectorArray<float,4,ISAAC_VECTOR_ELEM> start_p;
            VectorArray<float,4,ISAAC_VECTOR_ELEM> end_p;
            VectorArray<float,3,ISAAC_VECTOR_ELEM> start;
            VectorArray<float,3,ISAAC_VECTOR_ELEM> end;
            VectorArray<int,3,ISAAC_VECTOR_ELEM> move;
            VectorArray<float,3,ISAAC_VECTOR_ELEM> move_f;
            clipping_struct clipping[ISAAC_VECTOR_ELEM];
            VectorArray<float,3,ISAAC_VECTOR_ELEM> vec;
            VectorArray<isaac_float,1,ISAAC_VECTOR_ELEM> l_scaled;
            VectorArray<isaac_float,1,ISAAC_VECTOR_ELEM> l;
            VectorArray<isaac_float,1,ISAAC_VECTOR_ELEM> max_size;
            VectorArray<float,3,ISAAC_VECTOR_ELEM> step_vec;
            VectorArray<float,3,ISAAC_VECTOR_ELEM> count_start;
            VectorArray<float,3,ISAAC_VECTOR_ELEM> local_size_f;
            VectorArray<float,3,ISAAC_VECTOR_ELEM> count_end;

            ISAAC_ELEM_ITERATE(e)
                pixel_f.data[e].value.x = isaac_float( pixel.data[e].value.x )/(isaac_float)framebuffer_size.value.x*isaac_float(2)-isaac_float(1);
            ISAAC_ELEM_ITERATE(e)
                pixel_f.data[e].value.y = isaac_float( pixel.data[e].value.y )/(isaac_float)framebuffer_size.value.y*isaac_float(2)-isaac_float(1);

            ISAAC_ELEM_ITERATE(e)
                start_p.data[e].value.x = pixel_f.data[e].value.x*ISAAC_Z_NEAR;
            ISAAC_ELEM_ITERATE(e)
                start_p.data[e].value.y = pixel_f.data[e].value.y*ISAAC_Z_NEAR;
            ISAAC_ELEM_ITERATE(e)
                start_p.data[e].value.z = -1.0f*ISAAC_Z_NEAR;
            ISAAC_ELEM_ITERATE(e)
                start_p.data[e].value.w = 1.0f*ISAAC_Z_NEAR;

            ISAAC_ELEM_ITERATE(e)
                end_p.data[e].value.x = pixel_f.data[e].value.x*ISAAC_Z_FAR;
            ISAAC_ELEM_ITERATE(e)
                end_p.data[e].value.y = pixel_f.data[e].value.y*ISAAC_Z_FAR;
            ISAAC_ELEM_ITERATE(e)
                end_p.data[e].value.z = 1.0f*ISAAC_Z_FAR;
            ISAAC_ELEM_ITERATE(e)
                end_p.data[e].value.w = 1.0f*ISAAC_Z_FAR;

            ISAAC_ELEM_ITERATE(e)
                start.data[e].value.x = isaac_inverse_d[ 0] * start_p.data[e].value.x + isaac_inverse_d[ 4] * start_p.data[e].value.y +  isaac_inverse_d[ 8] * start_p.data[e].value.z + isaac_inverse_d[12] * start_p.data[e].value.w;
            ISAAC_ELEM_ITERATE(e)
                start.data[e].value.y = isaac_inverse_d[ 1] * start_p.data[e].value.x + isaac_inverse_d[ 5] * start_p.data[e].value.y +  isaac_inverse_d[ 9] * start_p.data[e].value.z + isaac_inverse_d[13] * start_p.data[e].value.w;
            ISAAC_ELEM_ITERATE(e)
                start.data[e].value.z = isaac_inverse_d[ 2] * start_p.data[e].value.x + isaac_inverse_d[ 6] * start_p.data[e].value.y +  isaac_inverse_d[10] * start_p.data[e].value.z + isaac_inverse_d[14] * start_p.data[e].value.w;

            ISAAC_ELEM_ITERATE(e)
                end.data[e].value.x =   isaac_inverse_d[ 0] *   end_p.data[e].value.x + isaac_inverse_d[ 4] *   end_p.data[e].value.y +  isaac_inverse_d[ 8] *   end_p.data[e].value.z + isaac_inverse_d[12] *   end_p.data[e].value.w;
            ISAAC_ELEM_ITERATE(e)
                end.data[e].value.y =   isaac_inverse_d[ 1] *   end_p.data[e].value.x + isaac_inverse_d[ 5] *   end_p.data[e].value.y +  isaac_inverse_d[ 9] *   end_p.data[e].value.z + isaac_inverse_d[13] *   end_p.data[e].value.w;
            ISAAC_ELEM_ITERATE(e)
                end.data[e].value.z =   isaac_inverse_d[ 2] *   end_p.data[e].value.x + isaac_inverse_d[ 6] *   end_p.data[e].value.y +  isaac_inverse_d[10] *   end_p.data[e].value.z + isaac_inverse_d[14] *   end_p.data[e].value.w;
            ISAAC_ELEM_ITERATE(e)
                max_size.array[e] = isaac_size_d[0].max_global_size_scaled / 2.0f;

                //scale to globale grid size
                start = start * max_size;
                  end =   end * max_size;

            for (isaac_int i = 0; i < input_clipping.count; i++)
            {
                ISAAC_ELEM_ITERATE(e)
                    clipping[e].elem[i].position = input_clipping.elem[i].position * max_size.array[e];
                ISAAC_ELEM_ITERATE(e)
                    clipping[e].elem[i].normal   = input_clipping.elem[i].normal;
            }

                //move to local (scaled) grid
            ISAAC_ELEM_ITERATE(e)
                move.data[e].value.x = isaac_int(isaac_size_d[0].global_size_scaled.value.x) / isaac_int(2) - isaac_int(isaac_size_d[0].position_scaled.value.x);
            ISAAC_ELEM_ITERATE(e)
                move.data[e].value.y = isaac_int(isaac_size_d[0].global_size_scaled.value.y) / isaac_int(2) - isaac_int(isaac_size_d[0].position_scaled.value.y);
            ISAAC_ELEM_ITERATE(e)
                move.data[e].value.z = isaac_int(isaac_size_d[0].global_size_scaled.value.z) / isaac_int(2) - isaac_int(isaac_size_d[0].position_scaled.value.z);

            ISAAC_ELEM_ITERATE(e)
                move_f.data[e].value.x = isaac_float(move.data[e].value.x);
            ISAAC_ELEM_ITERATE(e)
                move_f.data[e].value.y = isaac_float(move.data[e].value.y);
            ISAAC_ELEM_ITERATE(e)
                move_f.data[e].value.z = isaac_float(move.data[e].value.z);

            start = start + move_f;
              end = end   + move_f;
            for (isaac_int i = 0; i < input_clipping.count; i++)
                ISAAC_ELEM_ITERATE(e)
                    clipping[e].elem[i].position = clipping[e].elem[i].position + move_f.data[e];

            vec = end - start;
            ISAAC_ELEM_ITERATE(e)
                l_scaled.array[e] = sqrt( vec.data[e].value.x * vec.data[e].value.x + vec.data[e].value.y * vec.data[e].value.y + vec.data[e].value.z * vec.data[e].value.z );

            start = start / scale;
              end =   end / scale;
            for (isaac_int i = 0; i < input_clipping.count; i++)
                ISAAC_ELEM_ITERATE(e)
                    clipping[e].elem[i].position = clipping[e].elem[i].position / scale;

            vec = end - start;
            ISAAC_ELEM_ITERATE(e)
                l.array[e] = sqrt( vec.data[e].value.x * vec.data[e].value.x + vec.data[e].value.y * vec.data[e].value.y + vec.data[e].value.z * vec.data[e].value.z );

            step_vec = vec / l * step;
            count_start =  - start / step_vec;
            ISAAC_ELEM_ITERATE(e)
                local_size_f.data[e].value.x = isaac_float(isaac_size_d[0].local_size.value.x);
            ISAAC_ELEM_ITERATE(e)
                local_size_f.data[e].value.y = isaac_float(isaac_size_d[0].local_size.value.y);
            ISAAC_ELEM_ITERATE(e)
                local_size_f.data[e].value.z = isaac_float(isaac_size_d[0].local_size.value.z);

            count_end = ( local_size_f - start ) / step_vec;

                //count_start shall have the smaller values
            ISAAC_ELEM_ITERATE(e)
                ISAAC_SWITCH_IF_SMALLER( count_end.data[e].value.x, count_start.data[e].value.x )
            ISAAC_ELEM_ITERATE(e)
                ISAAC_SWITCH_IF_SMALLER( count_end.data[e].value.y, count_start.data[e].value.y )
            ISAAC_ELEM_ITERATE(e)
                ISAAC_SWITCH_IF_SMALLER( count_end.data[e].value.z, count_start.data[e].value.z )

                //calc intersection of all three super planes and save in [count_start.x ; count_end.x]
            ISAAC_ELEM_ITERATE(e)
                count_start.data[e].value.x = ISAAC_MAX( ISAAC_MAX( count_start.data[e].value.x, count_start.data[e].value.y ), count_start.data[e].value.z );
            ISAAC_ELEM_ITERATE(e)
                  count_end.data[e].value.x = ISAAC_MIN( ISAAC_MIN(   count_end.data[e].value.x,   count_end.data[e].value.y ),   count_end.data[e].value.z );
            ISAAC_ELEM_ITERATE(e)
                if ( count_start.data[e].value.x > count_end.data[e].value.x)
                {
                    if ( ISAAC_CHECK_FINISH(finish) )
                        ISAAC_SET_COLOR( pixels[pixel.data[e].value.x + pixel.data[e].value.y * framebuffer_size.value.x], color.data[e] )
                    finish.array[e] = true;
                }

            ISAAC_ELEM_ALL_TRUE_RETURN( finish.array )

            VectorArray<isaac_int,1,ISAAC_VECTOR_ELEM> first;
            VectorArray<isaac_int,1,ISAAC_VECTOR_ELEM> last;
            VectorArray<isaac_float,3,ISAAC_VECTOR_ELEM> pos;
            VectorArray<isaac_int,3,ISAAC_VECTOR_ELEM> coord;
            VectorArray<isaac_float,1,ISAAC_VECTOR_ELEM> d;
            VectorArray<isaac_float,1,ISAAC_VECTOR_ELEM> intersection_step;

            ISAAC_ELEM_ITERATE(e)
                first.array[e] = isaac_int( floor(count_start.data[e].value.x) );
            ISAAC_ELEM_ITERATE(e)
                last.array[e] = isaac_int( ceil(count_end.data[e].value.x) );

                //Moving last and first until their points are valid
            ISAAC_ELEM_ITERATE(e)
                pos.data[e] = start.data[e] + step_vec.data[e] * isaac_float(last.array[e]);
            ISAAC_ELEM_ITERATE(e)
                coord.data[e].value.x = isaac_int(floor(pos.data[e].value.x));
            ISAAC_ELEM_ITERATE(e)
                coord.data[e].value.y = isaac_int(floor(pos.data[e].value.y));
            ISAAC_ELEM_ITERATE(e)
                coord.data[e].value.z = isaac_int(floor(pos.data[e].value.z));
            ISAAC_ELEM_ITERATE(e)
                while ( (ISAAC_FOR_EACH_DIM_TWICE(3, coord.data[e], >= isaac_size_d[0].local_size, || )
                         ISAAC_FOR_EACH_DIM      (3, coord.data[e], < 0 || ) 0 ) && first.array[e] <= last.array[e])
                {
                    last.array[e]--;
                    pos.data[e] = start.data[e] + step_vec.data[e] * isaac_float(last.array[e]);
                    coord.data[e].value.x = isaac_int(floor(pos.data[e].value.x));
                    coord.data[e].value.y = isaac_int(floor(pos.data[e].value.y));
                    coord.data[e].value.z = isaac_int(floor(pos.data[e].value.z));
                }
            ISAAC_ELEM_ITERATE(e)
                pos.data[e] = start.data[e] + step_vec.data[e] * isaac_float(first.array[e]);
            ISAAC_ELEM_ITERATE(e)
                coord.data[e].value.x = isaac_int(floor(pos.data[e].value.x));
            ISAAC_ELEM_ITERATE(e)
                coord.data[e].value.y = isaac_int(floor(pos.data[e].value.y));
            ISAAC_ELEM_ITERATE(e)
                coord.data[e].value.z = isaac_int(floor(pos.data[e].value.z));
            ISAAC_ELEM_ITERATE(e)
                while ( (ISAAC_FOR_EACH_DIM_TWICE(3, coord.data[e], >= isaac_size_d[0].local_size, || )
                         ISAAC_FOR_EACH_DIM      (3, coord.data[e], < 0 || ) 0 ) && first.array[e] <= last.array[e])
                {
                    first.array[e]++;
                    pos.data[e] = start.data[e] + step_vec.data[e] * isaac_float(first.array[e]);
                    coord.data[e].value.x = isaac_int(floor(pos.data[e].value.x));
                    coord.data[e].value.y = isaac_int(floor(pos.data[e].value.y));
                    coord.data[e].value.z = isaac_int(floor(pos.data[e].value.z));
                }

            //Extra clipping
            for (isaac_int i = 0; i < input_clipping.count; i++)
            {
                ISAAC_ELEM_ITERATE(e)
                    d.array[e] = step_vec.data[e].value.x * clipping[e].elem[i].normal.value.x
                               + step_vec.data[e].value.y * clipping[e].elem[i].normal.value.y
                               + step_vec.data[e].value.z * clipping[e].elem[i].normal.value.z;
                ISAAC_ELEM_ITERATE(e)
                    intersection_step.array[e] = ( clipping[e].elem[i].position.value.x * clipping[e].elem[i].normal.value.x
                                                 + clipping[e].elem[i].position.value.y * clipping[e].elem[i].normal.value.y
                                                 + clipping[e].elem[i].position.value.z * clipping[e].elem[i].normal.value.z
                                                 -                start.data[e].value.x * clipping[e].elem[i].normal.value.x
                                                 -                start.data[e].value.y * clipping[e].elem[i].normal.value.y
                                                 -                start.data[e].value.z * clipping[e].elem[i].normal.value.z ) / d.array[e];
                ISAAC_ELEM_ITERATE(e)
                    if (d.array[e] > 0)
                    {
                        if ( last.array[e] < intersection_step.array[e] )
                        {
                            if ( ISAAC_CHECK_FINISH(finish) )
                                ISAAC_SET_COLOR( pixels[pixel.data[e].value.x + pixel.data[e].value.y * framebuffer_size.value.x], color.data[e] )
                            finish.array[e] = true;
                        }
                        if ( first.array[e] < intersection_step.array[e] )
                            first.array[e] = ceil( intersection_step.array[e] );
                    }
                    else
                    {
                        if ( first.array[e] > intersection_step.array[e] )
                        {
                            if ( ISAAC_CHECK_FINISH(finish) )
                                ISAAC_SET_COLOR( pixels[pixel.data[e].value.x + pixel.data[e].value.y * framebuffer_size.value.x], color.data[e] )
                            finish.array[e] = true;
                        }
                        if ( last.array[e] > intersection_step.array[e] )
                            last.array[e] = floor( intersection_step.array[e] );
                    }
            }
            ISAAC_ELEM_ALL_TRUE_RETURN( finish.array )

            VectorArray<isaac_float,1,ISAAC_VECTOR_ELEM> min_size;
            VectorArray<isaac_float,1,ISAAC_VECTOR_ELEM> factor;
            VectorArray<isaac_float,4,ISAAC_VECTOR_ELEM> value;
            VectorArray<isaac_float,1,ISAAC_VECTOR_ELEM> oma;
            VectorArray<isaac_float,4,ISAAC_VECTOR_ELEM> color_add;

                //Starting the main loop
            ISAAC_ELEM_ITERATE(e)
                min_size.array[e] = ISAAC_MIN(
                    int(isaac_size_d[0].global_size.value.x), ISAAC_MIN (
                    int(isaac_size_d[0].global_size.value.y),
                    int(isaac_size_d[0].global_size.value.z) ) );
            ISAAC_ELEM_ITERATE(e)
                factor.array[e] = step / /*isaac_size_d[0].max_global_size*/ min_size.array[e] * isaac_float(2) * l.array[e]/l_scaled.array[e];
            VectorArray<bool,1,ISAAC_VECTOR_ELEM> loop_finish = finish;
            for (VectorArray<isaac_int,1,ISAAC_VECTOR_ELEM> i = first;; i = i + 1)
            {
                ISAAC_ELEM_ITERATE(e)
                    if (ISAAC_CHECK_FINISH(loop_finish) && i.array[e] > last.array[e])
                    {
                        loop_finish.array[e] = true;
                    }
                ISAAC_ELEM_ALL_TRUE_BREAK( loop_finish.array )
                ISAAC_ELEM_ITERATE(e)
                    pos.data[e] = start.data[e] + step_vec.data[e] * isaac_float(i.array[e]);
                value = Vector<float,4>({0,0,0,0});
                isaac_for_each_with_mpl_params
                (
                    sources,
                    merge_source_iterator
                    <
                        Ttransfer_size,
                        TFilter,
                        TInterpolation,
                        TIsoSurface
                    >(),
                    value, //array
                    pos, //array
                    isaac_size_d[0].local_size,
                    transferArray,
                    sourceWeight,
                    pointerArray,
                    step_vec, //array
                    step,
                    scale,
                    loop_finish //array
                );
                if (TIsoSurface)
                {
                    ISAAC_ELEM_ITERATE(e)
                        if (loop_finish.array[e] && value.data[e].value.w < isaac_float(0))
                        {
                            value.data[e].value.w = isaac_float(1);
                            color.data[e] = value.data[e];
                            loop_finish.array[e] = true;
                        }
                    ISAAC_ELEM_ALL_TRUE_BREAK( loop_finish.array )
                }
                else
                {
                    ISAAC_ELEM_ITERATE(e)
                        oma.array[e] = isaac_float(1) - color.data[e].value.w;
                    ISAAC_ELEM_ITERATE(e)
                        value.data[e] = value.data[e] * factor.array[e];
                    ISAAC_ELEM_ITERATE(e)
                        color_add.data[e].value.x = oma.array[e] * value.data[e].value.x; // * value.value.w does merge_source_iterator
                    ISAAC_ELEM_ITERATE(e)
                        color_add.data[e].value.y = oma.array[e] * value.data[e].value.y; // * value.value.w does merge_source_iterator
                    ISAAC_ELEM_ITERATE(e)
                        color_add.data[e].value.z = oma.array[e] * value.data[e].value.z; // * value.value.w does merge_source_iterator
                    ISAAC_ELEM_ITERATE(e)
                        color_add.data[e].value.w = oma.array[e] * value.data[e].value.w;
                    ISAAC_ELEM_ITERATE(e)
                        if ( ISAAC_CHECK_FINISH(loop_finish) )
                            color.data[e] = color.data[e] + color_add.data[e];
                    ISAAC_ELEM_ITERATE(e)
                        if (color.data[e].value.w > isaac_float(0.99))
                            loop_finish.array[e] = true;
                    ISAAC_ELEM_ALL_TRUE_BREAK( loop_finish.array )
                }
            }
            #if ISAAC_SHOWBORDER == 1
                ISAAC_ELEM_ITERATE(e)
                    if (color.data[e].value.w <= isaac_float(0.99))
                    {
                        oma.array[e] = isaac_float(1) - color.data[e].value.w;
                        color_add.data[e].value.x = 0;
                        color_add.data[e].value.y = 0;
                        color_add.data[e].value.z = 0;
                        color_add.data[e].value.w = oma.array[e] * factor.array[e] * isaac_float(10);
                        color.data[e] = color.data[e] + color_add.data[e];
                    }
            #endif
            ISAAC_ELEM_ITERATE(e)
                if ( ISAAC_CHECK_FINISH(finish) )
                    ISAAC_SET_COLOR( pixels[pixel.data[e].value.x + pixel.data[e].value.y * framebuffer_size.value.x], color.data[e] )
        }
#if ISAAC_ALPAKA == 1
    };
#endif

template <
    typename TSimDim,
    typename TSourceList,
    typename TTransferArray,
    typename TSourceWeight,
    typename TPointerArray,
    typename TFilter,
    typename TFramebuffer,
    size_t TTransfer_size,
    typename TScale,
#if ISAAC_ALPAKA == 1
    typename TAccDim,
    typename TAcc,
    typename TStream,
    typename TFunctionChain,
#endif
    int N
>
struct IsaacRenderKernelCaller
{
    inline static void call(
#if ISAAC_ALPAKA == 1
        TStream stream,
#endif
        TFramebuffer framebuffer,
        const Vector<size_t,2>& framebuffer_size,
        const Vector<uint32_t,2>& framebuffer_start,
        const TSourceList& sources,
        const isaac_float& step,
        const Vector<float,4>& background_color,
        const TTransferArray& transferArray,
        const TSourceWeight& sourceWeight,
        const TPointerArray& pointerArray,
        IceTInt const * const readback_viewport,
        const isaac_int interpolation,
        const isaac_int iso_surface,
        const TScale& scale,
        const clipping_struct& clipping
    )
    {
        if (sourceWeight.value[ mpl::size< TSourceList >::type::value - N] == isaac_float(0) )
            IsaacRenderKernelCaller
            <
                TSimDim,
                TSourceList,
                TTransferArray,
                TSourceWeight,
                TPointerArray,
                typename mpl::push_back< TFilter, mpl::false_ >::type,
                TFramebuffer,
                TTransfer_size,
                TScale,
#if ISAAC_ALPAKA == 1
                TAccDim,
                TAcc,
                TStream,
                TFunctionChain,
#endif
                N - 1
            >
            ::call(
#if ISAAC_ALPAKA == 1
                stream,
#endif
                framebuffer,
                framebuffer_size,
                framebuffer_start,
                sources,
                step,
                background_color,
                transferArray,
                sourceWeight,
                pointerArray,
                readback_viewport,
                interpolation,
                iso_surface,
                scale,
                clipping
            );
    else
            IsaacRenderKernelCaller
            <
                TSimDim,
                TSourceList,
                TTransferArray,
                TSourceWeight,
                TPointerArray,
                typename mpl::push_back< TFilter, mpl::true_ >::type,
                TFramebuffer,
                TTransfer_size,
                TScale,
#if ISAAC_ALPAKA == 1
                TAccDim,
                TAcc,
                TStream,
                TFunctionChain,
#endif
                N - 1
            >
            ::call(
#if ISAAC_ALPAKA == 1
                stream,
#endif
                framebuffer,
                framebuffer_size,
                framebuffer_start,
                sources,
                step,
                background_color,
                transferArray,
                sourceWeight,
                pointerArray,
                readback_viewport,
                interpolation,
                iso_surface,
                scale,
                clipping
            );
    }
};

template <
    typename TSimDim,
    typename TSourceList,
    typename TTransferArray,
    typename TSourceWeight,
    typename TPointerArray,
    typename TFilter,
    typename TFramebuffer,
    size_t TTransfer_size,
    typename TScale
#if ISAAC_ALPAKA == 1
    ,typename TAccDim
    ,typename TAcc
    ,typename TStream
    ,typename TFunctionChain
#endif
>
struct IsaacRenderKernelCaller
<
    TSimDim,
    TSourceList,
    TTransferArray,
    TSourceWeight,
    TPointerArray,
    TFilter,
    TFramebuffer,
    TTransfer_size,
    TScale,
#if ISAAC_ALPAKA == 1
    TAccDim,
    TAcc,
    TStream,
    TFunctionChain,
#endif
    0 //<-- spezialisation
>
{
    inline static void call(
#if ISAAC_ALPAKA == 1
        TStream stream,
#endif
        TFramebuffer framebuffer,
        const Vector<size_t,2>& framebuffer_size,
        const Vector<uint32_t,2>& framebuffer_start,
        const TSourceList& sources,
        const isaac_float& step,
        const Vector<float,4>& background_color,
        const TTransferArray& transferArray,
        const TSourceWeight& sourceWeight,
        const TPointerArray& pointerArray,
        IceTInt const * const readback_viewport,
        const isaac_int interpolation,
        const isaac_int iso_surface,
        const TScale& scale,
        const clipping_struct& clipping
    )
    {
        Vector<size_t,2> block_size=
        {
            size_t(8),
            size_t(16)
        };
        Vector<size_t,2> grid_size=
        {
            size_t((readback_viewport[2]+block_size.value.x-1)/block_size.value.x + ISAAC_VECTOR_ELEM - 1)/size_t(ISAAC_VECTOR_ELEM),
            size_t((readback_viewport[3]+block_size.value.y-1)/block_size.value.y)
        };
        #if ISAAC_ALPAKA == 1
#if ALPAKA_ACC_GPU_CUDA_ENABLED == 1
            if ( mpl::not_<boost::is_same<TAcc, alpaka::acc::AccGpuCudaRt<TAccDim, size_t> > >::value )
#endif
            {
                grid_size.value.x = size_t(readback_viewport[2] + ISAAC_VECTOR_ELEM - 1)/size_t(ISAAC_VECTOR_ELEM);
                grid_size.value.y = size_t(readback_viewport[3]);
                block_size.value.x = size_t(1);
                block_size.value.y = size_t(1);
            }
            const alpaka::Vec<TAccDim, size_t> threads (size_t(1), size_t(1), size_t(ISAAC_VECTOR_ELEM));
            const alpaka::Vec<TAccDim, size_t> blocks  (size_t(1), block_size.value.y, block_size.value.x);
            const alpaka::Vec<TAccDim, size_t> grid    (size_t(1), grid_size.value.y, grid_size.value.x);
            auto const workdiv(alpaka::workdiv::WorkDivMembers<TAccDim, size_t>(grid,blocks,threads));
            #define ISAAC_KERNEL_START \
            { \
                isaacRenderKernel \
                < \
                    TSimDim, \
                    TSourceList, \
                    TTransferArray, \
                    TSourceWeight, \
                    TPointerArray, \
                    TFilter, \
                    TTransfer_size,
            #define ISAAC_KERNEL_END \
                    ,TScale \
                > \
                kernel; \
                auto const instance \
                ( \
                    alpaka::exec::create<TAcc> \
                    ( \
                        workdiv, \
                        kernel, \
                        alpaka::mem::view::getPtrNative(framebuffer), \
                        framebuffer_size, \
                        framebuffer_start, \
                        sources, \
                        step, \
                        background_color, \
                        transferArray, \
                        sourceWeight, \
                        pointerArray, \
                        scale, \
                        clipping \
                    ) \
                ); \
                alpaka::stream::enqueue(stream, instance); \
            }
        #else
            dim3 block (block_size.value.x, block_size.value.y);
            dim3 grid  (grid_size.value.x, grid_size.value.y);
            #define ISAAC_KERNEL_START \
                isaacRenderKernel \
                < \
                    TSimDim, \
                    TSourceList, \
                    TTransferArray, \
                    TSourceWeight, \
                    TPointerArray, \
                    TFilter, \
                    TTransfer_size,
            #define ISAAC_KERNEL_END \
                > \
                <<<grid, block>>> \
                ( \
                    framebuffer, \
                    framebuffer_size, \
                    framebuffer_start, \
                    sources, \
                    step, \
                    background_color, \
                    transferArray, \
                    sourceWeight, \
                    pointerArray, \
                    scale, \
                    clipping \
                );

        #endif
        if (interpolation)
        {
            if (iso_surface)
                ISAAC_KERNEL_START
                    1,
                    1
                ISAAC_KERNEL_END
            else
                ISAAC_KERNEL_START
                    1,
                    0
                ISAAC_KERNEL_END
        }
        else
        {
            if (iso_surface)
                ISAAC_KERNEL_START
                    0,
                    1
                ISAAC_KERNEL_END
            else
                ISAAC_KERNEL_START
                    0,
                    0
                ISAAC_KERNEL_END
        }
        #undef ISAAC_KERNEL_START
        #undef ISAAC_KERNEL_END
    }
};

template <int N>
struct dest_array_struct
{
    isaac_int nr[N];
};

template
<
    int count,
    typename TDest
>
#if ISAAC_ALPAKA == 1
    struct updateFunctorChainPointerKernel
    {
        template <typename TAcc__>
        ALPAKA_FN_ACC void operator()(
            TAcc__ const &acc,
#else
        __global__ void updateFunctorChainPointerKernel(
#endif
            isaac_functor_chain_pointer_N * const functor_chain_choose_d,
            isaac_functor_chain_pointer_N const * const functor_chain_d,
            TDest dest)
#if ISAAC_ALPAKA == 1
        const
#endif
        {
            for (int i = 0; i < count; i++)
                functor_chain_choose_d[i] = functor_chain_d[dest.nr[i]];
        }
#if ISAAC_ALPAKA == 1
    };
#endif

template
<
    typename TSource
>
#if ISAAC_ALPAKA == 1
    struct updateBufferKernel
    {
        template <typename TAcc__>
        ALPAKA_FN_ACC void operator()(
            TAcc__ const &acc,
#else
        __global__ void updateBufferKernel(
#endif
            const TSource source,
            void * const pointer,
            const Vector<int,3> local_size)
#if ISAAC_ALPAKA == 1
        const
#endif
        {
            #if ISAAC_ALPAKA == 1
                auto threadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
                Vector<int,3> dest =
                {
                    isaac_int(threadIdx[1]),
                    isaac_int(threadIdx[2]),
                    0
                };
            #else
                Vector<int,3> dest =
                {
                    isaac_int(threadIdx.x + blockIdx.x * blockDim.x),
                    isaac_int(threadIdx.y + blockIdx.y * blockDim.y),
                    0
                };
            #endif
            Vector<int,3> coord = dest;
            coord.value.x -= ISAAC_GUARD_SIZE;
            coord.value.y -= ISAAC_GUARD_SIZE;
            if ( ISAAC_FOR_EACH_DIM_TWICE(2, dest, >= local_size, + 2 * ISAAC_GUARD_SIZE || ) 0 )
                return;
            Vector<float, TSource::feature_dim >* ptr = (Vector<float, TSource::feature_dim >*)(pointer);
            if (TSource::has_guard)
            {
                coord.value.z = -ISAAC_GUARD_SIZE;
                for (;dest.value.z < local_size.value.z + 2 * ISAAC_GUARD_SIZE; dest.value.z++)
                {
                    ptr[dest.value.x + dest.value.y * (local_size.value.x + 2 * ISAAC_GUARD_SIZE) + dest.value.z * ( (local_size.value.x + 2 * ISAAC_GUARD_SIZE) * (local_size.value.y + 2 * ISAAC_GUARD_SIZE) )] = source[coord];
                    coord.value.z++;
                }
            }
            else
            {
                if (coord.value.x < 0)
                    coord.value.x = 0;
                if (coord.value.x >= local_size.value.x)
                    coord.value.x = local_size.value.x-1;
                if (coord.value.y < 0)
                    coord.value.y = 0;
                if (coord.value.y >= local_size.value.y)
                    coord.value.y = local_size.value.y-1;
                coord.value.z = 0;
                for (; dest.value.z < ISAAC_GUARD_SIZE; dest.value.z++)
                    ptr[dest.value.x + dest.value.y * (local_size.value.x + 2 * ISAAC_GUARD_SIZE) + dest.value.z * ( (local_size.value.x + 2 * ISAAC_GUARD_SIZE) * (local_size.value.y + 2 * ISAAC_GUARD_SIZE) )] = source[coord];
                for (;dest.value.z < local_size.value.z + ISAAC_GUARD_SIZE - 1; dest.value.z++)
                {
                    ptr[dest.value.x + dest.value.y * (local_size.value.x + 2 * ISAAC_GUARD_SIZE) + dest.value.z * ( (local_size.value.x + 2 * ISAAC_GUARD_SIZE) * (local_size.value.y + 2 * ISAAC_GUARD_SIZE) )] = source[coord];
                    coord.value.z++;
                }
                for (;dest.value.z < local_size.value.z + 2 * ISAAC_GUARD_SIZE; dest.value.z++)
                    ptr[dest.value.x + dest.value.y * (local_size.value.x + 2 * ISAAC_GUARD_SIZE) + dest.value.z * ( (local_size.value.x + 2 * ISAAC_GUARD_SIZE) * (local_size.value.y + 2 * ISAAC_GUARD_SIZE) )] = source[coord];
            }
        }
#if ISAAC_ALPAKA == 1
    };
#endif

template
<
    typename TSource
>
#if ISAAC_ALPAKA == 1
    struct minMaxKernel
    {
        template <typename TAcc__>
        ALPAKA_FN_ACC void operator()(
            TAcc__ const &acc,
#else
        __global__ void minMaxKernel(
#endif
            const TSource source,
            const int nr,
            minmax_struct * const result,
            const Vector<int,3> local_size,
            void const * const pointer)
#if ISAAC_ALPAKA == 1
        const
#endif
        {
            #if ISAAC_ALPAKA == 1
                auto threadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
                Vector<int,3> coord =
                {
                    isaac_int(threadIdx[1]),
                    isaac_int(threadIdx[2]),
                    0
                };
            #else
                Vector<int,3> coord =
                {
                    isaac_int(threadIdx.x + blockIdx.x * blockDim.x),
                    isaac_int(threadIdx.y + blockIdx.y * blockDim.y),
                    0
                };
            #endif
            if ( ISAAC_FOR_EACH_DIM_TWICE(2, coord, >= local_size, || ) 0 )
                return;
            isaac_float min =  FLT_MAX;
            isaac_float max = -FLT_MAX;
            for (;coord.value.z < local_size.value.z; coord.value.z++)
            {
                Vector<float, TSource::feature_dim > data;
                if (TSource::persistent)
                    data = source[coord];
                else
                {
                    Vector<float, TSource::feature_dim >* ptr = (Vector<float, TSource::feature_dim >*)(pointer);
                    data = ptr[coord.value.x + ISAAC_GUARD_SIZE + (coord.value.y + ISAAC_GUARD_SIZE) * (local_size.value.x + 2 * ISAAC_GUARD_SIZE) + (coord.value.z + ISAAC_GUARD_SIZE) * ( (local_size.value.x + 2 * ISAAC_GUARD_SIZE) * (local_size.value.y + 2 * ISAAC_GUARD_SIZE) )];
                };
                isaac_float value = isaac_float(0);
                #if ISAAC_ALPAKA == 1 || defined(__CUDA_ARCH__)
                    if (TSource::feature_dim == 1)
                        value = reinterpret_cast<isaac_functor_chain_pointer_1>(isaac_function_chain_d[ nr ])( *(reinterpret_cast< Vector<float,1>* >(&data)), nr );
                    if (TSource::feature_dim == 2)
                        value = reinterpret_cast<isaac_functor_chain_pointer_2>(isaac_function_chain_d[ nr ])( *(reinterpret_cast< Vector<float,2>* >(&data)), nr );
                    if (TSource::feature_dim == 3)
                        value = reinterpret_cast<isaac_functor_chain_pointer_3>(isaac_function_chain_d[ nr ])( *(reinterpret_cast< Vector<float,3>* >(&data)), nr );
                    if (TSource::feature_dim == 4)
                        value = reinterpret_cast<isaac_functor_chain_pointer_4>(isaac_function_chain_d[ nr ])( *(reinterpret_cast< Vector<float,4>* >(&data)), nr );
                #endif
                if (value > max)
                    max = value;
                if (value < min)
                    min = value;
            }
            result[coord.value.x +  coord.value.y * local_size.value.x].min = min;
            result[coord.value.x +  coord.value.y * local_size.value.x].max = max;
        }
#if ISAAC_ALPAKA == 1
    };
#endif

} //namespace isaac;

#pragma GCC diagnostic pop
