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
    typename TSource,
    typename TPos,
    typename TPointerArray,
    typename TLocalSize,
    typename TScale
>
ISAAC_HOST_DEVICE_INLINE isaac_float get_value (
    const TSource& source,
    const TPos& pos,
    const TPointerArray& pointerArray,
    const TLocalSize& local_size,
    const TScale& scale
)
{
    Vector<float, TSource::feature_dim > data;
    Vector<float, TSource::feature_dim >* ptr = (Vector<float, TSource::feature_dim >*)(pointerArray.pointer[ NR::value ] );
    if (TInterpolation == 0)
    {
        Vector<int,3> coord =
        {
            isaac_int(pos.value.x),
            isaac_int(pos.value.y),
            isaac_int(pos.value.z)
        };
        if (TSource::persistent)
            data = source[coord];
        else
            data = ptr[coord.value.x + ISAAC_GUARD_SIZE + (coord.value.y + ISAAC_GUARD_SIZE) * (local_size.value.x + 2 * ISAAC_GUARD_SIZE) + (coord.value.z + ISAAC_GUARD_SIZE) * ( (local_size.value.x + 2 * ISAAC_GUARD_SIZE) * (local_size.value.y + 2 * ISAAC_GUARD_SIZE) )];
    }
    else
    {
        Vector<int,3> coord;
        Vector<float, TSource::feature_dim > data8[2][2][2];
        for (int x = 0; x < 2; x++)
            for (int y = 0; y < 2; y++)
                for (int z = 0; z < 2; z++)
                {
                    coord.value.x = isaac_int(x?ceil(pos.value.x):floor(pos.value.x));
                    coord.value.y = isaac_int(y?ceil(pos.value.y):floor(pos.value.y));
                    coord.value.z = isaac_int(z?ceil(pos.value.z):floor(pos.value.z));
                    if (!TSource::has_guard && TSource::persistent)
                    {
                        if ( isaac_uint(coord.value.x) >= local_size.value.x )
                            coord.value.x = isaac_int(x?floor(pos.value.x):ceil(pos.value.x));
                        if ( isaac_uint(coord.value.y) >= local_size.value.y )
                            coord.value.y = isaac_int(y?floor(pos.value.y):ceil(pos.value.y));
                        if ( isaac_uint(coord.value.z) >= local_size.value.z )
                            coord.value.z = isaac_int(z?floor(pos.value.z):ceil(pos.value.z));
                    }
                    if (TSource::persistent)
                        data8[x][y][z] = source[coord];
                    else
                        data8[x][y][z] = ptr[coord.value.x + ISAAC_GUARD_SIZE + (coord.value.y + ISAAC_GUARD_SIZE) * (local_size.value.x + 2 * ISAAC_GUARD_SIZE) + (coord.value.z + ISAAC_GUARD_SIZE) * ( (local_size.value.x + 2 * ISAAC_GUARD_SIZE) * (local_size.value.y + 2 * ISAAC_GUARD_SIZE) )];
                }
        Vector<float, 3 > pos_in_cube =
        {
            pos.value.x - floor(pos.value.x),
            pos.value.y - floor(pos.value.y),
            pos.value.z - floor(pos.value.z)
        };
        Vector<float, TSource::feature_dim > data4[2][2];
        for (int x = 0; x < 2; x++)
            for (int y = 0; y < 2; y++)
                data4[x][y] =
                    data8[x][y][0] * (isaac_float(1) - pos_in_cube.value.z) +
                    data8[x][y][1] * (                 pos_in_cube.value.z);
        Vector<float, TSource::feature_dim > data2[2];
        for (int x = 0; x < 2; x++)
            data2[x] =
                data4[x][0] * (isaac_float(1) - pos_in_cube.value.y) +
                data4[x][1] * (                 pos_in_cube.value.y);
        data =
            data2[0] * (isaac_float(1) - pos_in_cube.value.x) +
            data2[1] * (                 pos_in_cube.value.x);
    }
    isaac_float result = isaac_float(0);

    #if ISAAC_ALPAKA == 1 || defined(__CUDA_ARCH__)
        if (TSource::feature_dim == 1)
            result = reinterpret_cast<isaac_functor_chain_pointer_1>(isaac_function_chain_d[ NR::value ])( *(reinterpret_cast< Vector<float,1>* >(&data)), NR::value );
        if (TSource::feature_dim == 2)
            result = reinterpret_cast<isaac_functor_chain_pointer_2>(isaac_function_chain_d[ NR::value ])( *(reinterpret_cast< Vector<float,2>* >(&data)), NR::value );
        if (TSource::feature_dim == 3)
            result = reinterpret_cast<isaac_functor_chain_pointer_3>(isaac_function_chain_d[ NR::value ])( *(reinterpret_cast< Vector<float,3>* >(&data)), NR::value );
        if (TSource::feature_dim == 4)
            result = reinterpret_cast<isaac_functor_chain_pointer_4>(isaac_function_chain_d[ NR::value ])( *(reinterpret_cast< Vector<float,4>* >(&data)), NR::value );
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
        typename TFeedback,
        typename TStep,
        typename TStepLength,
        typename TScale
    >
    ISAAC_HOST_DEVICE_INLINE  void operator()(
        const NR& nr,
        const TSource& source,
        TColor& color,
        const TPos& pos,
        const TLocalSize& local_size,
        const TTransferArray& transferArray,
        const TSourceWeight& sourceWeight,
        const TPointerArray& pointerArray,
        TFeedback& feedback,
        const TStep& step,
        const TStepLength& stepLength,
        const TScale& scale
    ) const
    {
        if ( mpl::at_c< TFilter, NR::value >::type::value )
        {
            isaac_float result = get_value< TInterpolation, NR >( source, pos, pointerArray, local_size, scale );
            isaac_int lookup_value = isaac_int( round(result * isaac_float( Ttransfer_size ) ) );
            if (lookup_value < 0 )
                lookup_value = 0;
            if (lookup_value >= Ttransfer_size )
                lookup_value = Ttransfer_size - 1;
            Vector<float,4> value = transferArray.pointer[ NR::value ][ lookup_value ];
            if (TIsoSurface)
            {
                if (value.value.w >= isaac_float(0.5))
                {
                    Vector<float,3>  left = {-1, 0, 0};
                    left = left + pos;
                    if (!TSource::has_guard && TSource::persistent)
                        check_coord( left, local_size);
                    Vector<float,3> right = { 1, 0, 0};
                    right = right + pos;
                    if (!TSource::has_guard && TSource::persistent)
                        check_coord( right, local_size );
                    isaac_float d1;
                    if (TInterpolation)
                        d1 = right.value.x - left.value.x;
                    else
                        d1 = isaac_int(right.value.x) - isaac_int(left.value.x);

                    Vector<float,3>    up = { 0,-1, 0};
                    up = up + pos;
                    if (!TSource::has_guard && TSource::persistent)
                        check_coord( up, local_size );
                    Vector<float,3>  down = { 0, 1, 0};
                    down = down + pos;
                    if (!TSource::has_guard && TSource::persistent)
                        check_coord( down, local_size );
                    isaac_float d2;
                    if (TInterpolation)
                        d2 = down.value.y - up.value.y;
                    else
                        d2 = isaac_int(down.value.y) - isaac_int(up.value.y);

                    Vector<float,3> front = { 0, 0,-1};
                    front = front + pos;
                    if (!TSource::has_guard && TSource::persistent)
                        check_coord( front, local_size );
                    Vector<float,3>  back = { 0, 0, 1};
                    back = back + pos;
                    if (!TSource::has_guard && TSource::persistent)
                        check_coord( back, local_size );
                    isaac_float d3;
                    if (TInterpolation)
                        d3 = back.value.z - front.value.z;
                    else
                        d3 = isaac_int(back.value.z) - isaac_int(front.value.z);

                    Vector<float,3> gradient=
                    {
                        (get_value< TInterpolation, NR >( source, right, pointerArray, local_size, scale ) -
                         get_value< TInterpolation, NR >( source,  left, pointerArray, local_size, scale )) / d1,
                        (get_value< TInterpolation, NR >( source,  down, pointerArray, local_size, scale ) -
                         get_value< TInterpolation, NR >( source,    up, pointerArray, local_size, scale )) / d2,
                        (get_value< TInterpolation, NR >( source,  back, pointerArray, local_size, scale ) -
                         get_value< TInterpolation, NR >( source, front, pointerArray, local_size, scale )) / d3
                    };
                    isaac_float l = sqrt(
                        gradient.value.x * gradient.value.x +
                        gradient.value.y * gradient.value.y +
                        gradient.value.z * gradient.value.z
                    );
                    if (l == isaac_float(0))
                        color = value;
                    else
                    {
                        gradient = gradient / l;
                        Vector<float,3> light = step / stepLength;
                        isaac_float ac = fabs(
                            gradient.value.x * light.value.x +
                            gradient.value.y * light.value.y +
                            gradient.value.z * light.value.z );
                        #if ISAAC_SPECULAR == 1
                            color.value.x = value.value.x * ac + ac * ac * ac * ac;
                            color.value.y = value.value.y * ac + ac * ac * ac * ac;
                            color.value.z = value.value.z * ac + ac * ac * ac * ac;
                        #else
                            color.value.x = value.value.x * ac;
                            color.value.y = value.value.y * ac;
                            color.value.z = value.value.z * ac;
                        #endif
                    }
                    color.value.w = isaac_float(1);
                    feedback = 1;
                }
            }
            else
            {
                value.value.w *= sourceWeight.value[ NR::value ];
                color.value.x = color.value.x + value.value.x * value.value.w;
                color.value.y = color.value.y + value.value.y * value.value.w;
                color.value.z = color.value.z + value.value.z * value.value.w;
                color.value.w = color.value.w + value.value.w;
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
            Vector<uint32_t,2> pixel[ISAAC_VECTOR_ELEM];
            bool finish[ISAAC_VECTOR_ELEM];
#if ISAAC_ALPAKA == 1
            auto threadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            ISAAC_ELEM_ITERATE(e)
            {
                pixel[e].value.x = isaac_uint(threadIdx[2]) * isaac_uint(ISAAC_VECTOR_ELEM) + e;
                pixel[e].value.y = isaac_uint(threadIdx[1]);
#else
            ISAAC_ELEM_ITERATE(e)
            {
                pixel[e].value.x = isaac_uint(threadIdx.x + blockIdx.x * blockDim.x) * isaac_uint(ISAAC_VECTOR_ELEM) + e;
                pixel[e].value.y = isaac_uint(threadIdx.y + blockIdx.y * blockDim.y);
#endif
                finish[e] = false;
                pixel[e] = pixel[e] + framebuffer_start;
                if ( ISAAC_FOR_EACH_DIM_TWICE(2, pixel[e], >= framebuffer_size, || ) 0 )
                    finish[e] = true;
            }
            ISAAC_ELEM_ALL_TRUE_RETURN( finish )

            bool at_least_one[ISAAC_VECTOR_ELEM];
            Vector<float,4> color[ISAAC_VECTOR_ELEM];

            ISAAC_ELEM_ITERATE(e)
            {
                color[e] = background_color;
                at_least_one[e] = true;
                isaac_for_each_with_mpl_params( sources, check_no_source_iterator<TFilter>(), at_least_one[e] );
                if (!at_least_one[e])
                {
                    if (!finish[e])
                        ISAAC_SET_COLOR( pixels[pixel[e].value.x + pixel[e].value.y * framebuffer_size.value.x], color[e] )
                    finish[e] = true;
                }
            }
            ISAAC_ELEM_ALL_TRUE_RETURN( finish )

            Vector<float,2> pixel_f[ISAAC_VECTOR_ELEM];
            Vector<float,4> start_p[ISAAC_VECTOR_ELEM];
            Vector<float,4> end_p[ISAAC_VECTOR_ELEM];
            Vector<float,3> start[ISAAC_VECTOR_ELEM];
            Vector<float,3> end[ISAAC_VECTOR_ELEM];
            Vector<int,3> move[ISAAC_VECTOR_ELEM];
            Vector<float,3> move_f[ISAAC_VECTOR_ELEM];
            clipping_struct clipping[ISAAC_VECTOR_ELEM];
            Vector<float,3> vec[ISAAC_VECTOR_ELEM];
            isaac_float l_scaled[ISAAC_VECTOR_ELEM];
            isaac_float l[ISAAC_VECTOR_ELEM];
            Vector<float,3> step_vec[ISAAC_VECTOR_ELEM];
            Vector<float,3> count_start[ISAAC_VECTOR_ELEM];
            Vector<float,3> local_size_f[ISAAC_VECTOR_ELEM];
            Vector<float,3> count_end[ISAAC_VECTOR_ELEM];

            ISAAC_ELEM_ITERATE(e)
            {
                pixel_f[e].value.x = isaac_float( pixel[e].value.x )/(isaac_float)framebuffer_size.value.x*isaac_float(2)-isaac_float(1);
                pixel_f[e].value.y = isaac_float( pixel[e].value.y )/(isaac_float)framebuffer_size.value.y*isaac_float(2)-isaac_float(1);

                start_p[e].value.x = pixel_f[e].value.x*ISAAC_Z_NEAR;
                start_p[e].value.y = pixel_f[e].value.y*ISAAC_Z_NEAR;
                start_p[e].value.z = -1.0f*ISAAC_Z_NEAR;
                start_p[e].value.w = 1.0f*ISAAC_Z_NEAR;

                end_p[e].value.x = pixel_f[e].value.x*ISAAC_Z_FAR;
                end_p[e].value.y = pixel_f[e].value.y*ISAAC_Z_FAR;
                end_p[e].value.z = 1.0f*ISAAC_Z_FAR;
                end_p[e].value.w = 1.0f*ISAAC_Z_FAR;

                start[e].value.x = isaac_inverse_d[ 0] * start_p[e].value.x + isaac_inverse_d[ 4] * start_p[e].value.y +  isaac_inverse_d[ 8] * start_p[e].value.z + isaac_inverse_d[12] * start_p[e].value.w;
                start[e].value.y = isaac_inverse_d[ 1] * start_p[e].value.x + isaac_inverse_d[ 5] * start_p[e].value.y +  isaac_inverse_d[ 9] * start_p[e].value.z + isaac_inverse_d[13] * start_p[e].value.w;
                start[e].value.z = isaac_inverse_d[ 2] * start_p[e].value.x + isaac_inverse_d[ 6] * start_p[e].value.y +  isaac_inverse_d[10] * start_p[e].value.z + isaac_inverse_d[14] * start_p[e].value.w;

                end[e].value.x =   isaac_inverse_d[ 0] *   end_p[e].value.x + isaac_inverse_d[ 4] *   end_p[e].value.y +  isaac_inverse_d[ 8] *   end_p[e].value.z + isaac_inverse_d[12] *   end_p[e].value.w;
                end[e].value.y =   isaac_inverse_d[ 1] *   end_p[e].value.x + isaac_inverse_d[ 5] *   end_p[e].value.y +  isaac_inverse_d[ 9] *   end_p[e].value.z + isaac_inverse_d[13] *   end_p[e].value.w;
                end[e].value.z =   isaac_inverse_d[ 2] *   end_p[e].value.x + isaac_inverse_d[ 6] *   end_p[e].value.y +  isaac_inverse_d[10] *   end_p[e].value.z + isaac_inverse_d[14] *   end_p[e].value.w;
                isaac_float max_size = isaac_size_d[0].max_global_size_scaled / 2.0f;

                //scale to globale grid size
                start[e] = start[e] * max_size;
                  end[e] =   end[e] * max_size;

                for (isaac_int i = 0; i < input_clipping.count; i++)
                {
                    clipping[e].elem[i].position = input_clipping.elem[i].position * max_size;
                    clipping[e].elem[i].normal   = input_clipping.elem[i].normal;
                }

                //move to local (scaled) grid
                move[e].value.x = isaac_int(isaac_size_d[0].global_size_scaled.value.x) / isaac_int(2) - isaac_int(isaac_size_d[0].position_scaled.value.x);
                move[e].value.y = isaac_int(isaac_size_d[0].global_size_scaled.value.y) / isaac_int(2) - isaac_int(isaac_size_d[0].position_scaled.value.y);
                move[e].value.z = isaac_int(isaac_size_d[0].global_size_scaled.value.z) / isaac_int(2) - isaac_int(isaac_size_d[0].position_scaled.value.z);

                move_f[e].value.x = isaac_float(move[e].value.x);
                move_f[e].value.y = isaac_float(move[e].value.y);
                move_f[e].value.z = isaac_float(move[e].value.z);

                start[e] = start[e] + move_f[e];
                  end[e] =   end[e] + move_f[e];
                for (isaac_int i = 0; i < input_clipping.count; i++)
                    clipping[e].elem[i].position = clipping[e].elem[i].position + move_f[e];

                vec[e] = end[e] - start[e];
                l_scaled[e] = sqrt( vec[e].value.x * vec[e].value.x + vec[e].value.y * vec[e].value.y + vec[e].value.z * vec[e].value.z );

                start[e].value.x = start[e].value.x / scale.value.x;
                start[e].value.y = start[e].value.y / scale.value.y;
                start[e].value.z = start[e].value.z / scale.value.z;
                  end[e].value.x =   end[e].value.x / scale.value.x;
                  end[e].value.y =   end[e].value.y / scale.value.y;
                  end[e].value.z =   end[e].value.z / scale.value.z;
                for (isaac_int i = 0; i < input_clipping.count; i++)
                {
                    clipping[e].elem[i].position.value.x = clipping[e].elem[i].position.value.x / scale.value.x;
                    clipping[e].elem[i].position.value.y = clipping[e].elem[i].position.value.y / scale.value.y;
                    clipping[e].elem[i].position.value.z = clipping[e].elem[i].position.value.z / scale.value.z;
                }

                vec[e] = end[e] - start[e];
                l[e] = sqrt( vec[e].value.x * vec[e].value.x + vec[e].value.y * vec[e].value.y + vec[e].value.z * vec[e].value.z );

                step_vec[e] = vec[e] / l[e] * step;
                count_start[e] =  - start[e] / step_vec[e];
                local_size_f[e].value.x = isaac_float(isaac_size_d[0].local_size.value.x);
                local_size_f[e].value.y = isaac_float(isaac_size_d[0].local_size.value.y);
                local_size_f[e].value.z = isaac_float(isaac_size_d[0].local_size.value.z);

                count_end[e] = ( local_size_f[e] - start[e] ) / step_vec[e];

                //count_start shall have the smaller values
                ISAAC_SWITCH_IF_SMALLER( count_end[e].value.x, count_start[e].value.x )
                ISAAC_SWITCH_IF_SMALLER( count_end[e].value.y, count_start[e].value.y )
                ISAAC_SWITCH_IF_SMALLER( count_end[e].value.z, count_start[e].value.z )

                //calc intersection of all three super planes and save in [count_start.x ; count_end.x]
                count_start[e].value.x = ISAAC_MAX( ISAAC_MAX( count_start[e].value.x, count_start[e].value.y ), count_start[e].value.z );
                  count_end[e].value.x = ISAAC_MIN( ISAAC_MIN(   count_end[e].value.x,   count_end[e].value.y ),   count_end[e].value.z );
                if ( count_start[e].value.x > count_end[e].value.x)
                {
                    if (!finish[e])
                        ISAAC_SET_COLOR( pixels[pixel[e].value.x + pixel[e].value.y * framebuffer_size.value.x], color[e] )
                    finish[e] = true;
                }
            }
            ISAAC_ELEM_ALL_TRUE_RETURN( finish )

            isaac_int first[ISAAC_VECTOR_ELEM];
            isaac_int last[ISAAC_VECTOR_ELEM];
            Vector<float,3> pos[ISAAC_VECTOR_ELEM];
            Vector<int,3> coord[ISAAC_VECTOR_ELEM];
            isaac_float d[ISAAC_VECTOR_ELEM];
            isaac_float intersection_step[ISAAC_VECTOR_ELEM];

            ISAAC_ELEM_ITERATE(e)
            {
                first[e] = isaac_int( floor(count_start[e].value.x) );
                last[e] = isaac_int( ceil(count_end[e].value.x) );

                //Moving last and first until their points are valid
                pos[e] = start[e] + step_vec[e] * isaac_float(last[e]);
                coord[e].value.x = isaac_int(floor(pos[e].value.x));
                coord[e].value.y = isaac_int(floor(pos[e].value.y));
                coord[e].value.z = isaac_int(floor(pos[e].value.z));
                while ( (ISAAC_FOR_EACH_DIM_TWICE(3, coord[e], >= isaac_size_d[0].local_size, || )
                         ISAAC_FOR_EACH_DIM      (3, coord[e], < 0 || ) 0 ) && first[e] <= last[e])
                {
                    last[e]--;
                    pos[e] = start[e] + step_vec[e] * isaac_float(last[e]);
                    coord[e].value.x = isaac_int(floor(pos[e].value.x));
                    coord[e].value.y = isaac_int(floor(pos[e].value.y));
                    coord[e].value.z = isaac_int(floor(pos[e].value.z));
                }
                pos[e] = start[e] + step_vec[e] * isaac_float(first[e]);
                coord[e].value.x = isaac_int(floor(pos[e].value.x));
                coord[e].value.y = isaac_int(floor(pos[e].value.y));
                coord[e].value.z = isaac_int(floor(pos[e].value.z));
                while ( (ISAAC_FOR_EACH_DIM_TWICE(3, coord[e], >= isaac_size_d[0].local_size, || )
                         ISAAC_FOR_EACH_DIM      (3, coord[e], < 0 || ) 0 ) && first[e] <= last[e])
                {
                    first[e]++;
                    pos[e] = start[e] + step_vec[e] * isaac_float(first[e]);
                    coord[e].value.x = isaac_int(floor(pos[e].value.x));
                    coord[e].value.y = isaac_int(floor(pos[e].value.y));
                    coord[e].value.z = isaac_int(floor(pos[e].value.z));
                }

                //Extra clipping
                for (isaac_int i = 0; i < input_clipping.count; i++)
                {
                    d[e] = step_vec[e].value.x * clipping[e].elem[i].normal.value.x
                         + step_vec[e].value.y * clipping[e].elem[i].normal.value.y
                         + step_vec[e].value.z * clipping[e].elem[i].normal.value.z;
                    intersection_step[e] = ( clipping[e].elem[i].position.value.x * clipping[e].elem[i].normal.value.x
                                           + clipping[e].elem[i].position.value.y * clipping[e].elem[i].normal.value.y
                                           + clipping[e].elem[i].position.value.z * clipping[e].elem[i].normal.value.z
                                           -                     start[e].value.x * clipping[e].elem[i].normal.value.x
                                           -                     start[e].value.y * clipping[e].elem[i].normal.value.y
                                           -                     start[e].value.z * clipping[e].elem[i].normal.value.z ) / d[e];
                    if (d[e] > 0)
                    {
                        if ( last[e] < intersection_step[e] )
                        {
                            if (!finish[e])
                                ISAAC_SET_COLOR( pixels[pixel[e].value.x + pixel[e].value.y * framebuffer_size.value.x], color[e] )
                            finish[e] = true;
                        }
                        if ( first[e] < intersection_step[e] )
                            first[e] = ceil( intersection_step[e] );
                    }
                    else
                    {
                        if ( first[e] > intersection_step[e] )
                        {
                            if (!finish[e])
                                ISAAC_SET_COLOR( pixels[pixel[e].value.x + pixel[e].value.y * framebuffer_size.value.x], color[e] )
                            finish[e] = true;
                        }
                        if ( last[e] > intersection_step[e] )
                            last[e] = floor( intersection_step[e] );
                    }
                }
            }
            ISAAC_ELEM_ALL_TRUE_RETURN( finish )

            isaac_float min_size[ISAAC_VECTOR_ELEM];
            isaac_float factor[ISAAC_VECTOR_ELEM];
            Vector<float,4> value[ISAAC_VECTOR_ELEM];
            isaac_int result[ISAAC_VECTOR_ELEM];
            isaac_float oma[ISAAC_VECTOR_ELEM];
            Vector<float,4> color_add[ISAAC_VECTOR_ELEM];

            ISAAC_ELEM_ITERATE(e)
            {
                //Starting the main loop
                min_size[e] = ISAAC_MIN(
                    int(isaac_size_d[0].global_size.value.x), ISAAC_MIN (
                    int(isaac_size_d[0].global_size.value.y),
                    int(isaac_size_d[0].global_size.value.z) ) );
                factor[e] = step / /*isaac_size_d[0].max_global_size*/ min_size[e] * isaac_float(2) * l[e]/l_scaled[e];
                for (isaac_int i = first[e]; i <= last[e]; i++)
                {
                    pos[e] = start[e] + step_vec[e] * isaac_float(i);
                    value[e].value.x = 0;
                    value[e].value.y = 0;
                    value[e].value.z = 0;
                    value[e].value.w = 0;
                    result[e] = 0;
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
                        value[e],
                        pos[e],
                        isaac_size_d[0].local_size,
                        transferArray,
                        sourceWeight,
                        pointerArray,
                        result[e],
                        step_vec[e],
                        step,
                        scale
                    );
                    /*if ( mpl::size< TSourceList >::type::value > 1)
                        value = value / isaac_float( mpl::size< TSourceList >::type::value );*/
                    if (TIsoSurface)
                    {
                        if (result[e])
                        {
                            color[e] = value[e];
                            break;
                        }
                    }
                    else
                    {
                        oma[e] = isaac_float(1) - color[e].value.w;
                        value[e] = value[e] * factor[e];
                        color_add[e].value.x = oma[e] * value[e].value.x; // * value.value.w does merge_source_iterator
                        color_add[e].value.y = oma[e] * value[e].value.y; // * value.value.w does merge_source_iterator
                        color_add[e].value.z = oma[e] * value[e].value.z; // * value.value.w does merge_source_iterator
                        color_add[e].value.w = oma[e] * value[e].value.w;
                        color[e] = color[e] + color_add[e];
                        if (color[e].value.w > isaac_float(0.99))
                            break;
                    }
                }
                #if ISAAC_SHOWBORDER == 1
                    if (color[e].value.w <= isaac_float(0.99))
                    {
                        oma[e] = isaac_float(1) - color[e].value.w;
                        color_add[e].value.x = 0;
                        color_add[e].value.y = 0;
                        color_add[e].value.z = 0;
                        color_add[e].value.w = oma[e] * factor[e] * isaac_float(10);
                        };
                        color[e] = color[e] + color_add[e];
                    }
                #endif
                if (!finish[e])
                    ISAAC_SET_COLOR( pixels[pixel[e].value.x + pixel[e].value.y * framebuffer_size.value.x], color[e] )
            }
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
