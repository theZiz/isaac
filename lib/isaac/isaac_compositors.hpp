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

namespace isaac
{

struct DefaultCompositor
{
	DefaultCompositor( Vector<size_t,2> framebuffer_size ) {}
	~DefaultCompositor() {}
	static inline Vector<size_t,2> getCompositedbufferSize( Vector<size_t,2> framebuffer_size )
	{
		return framebuffer_size;
	}
	inline uint32_t* doCompositing(IceTImage* image)
	{
		return icetImageGetColorui(image[0]);
	}
};

template <typename TController>
class StereoCompositorSideBySide
{
	public:
		static inline Vector<size_t,2> getCompositedbufferSize( Vector<size_t,2> framebuffer_size )
		{
			Vector<size_t,2> compbuffer_size =
			{
				framebuffer_size.value.x * 2,
				framebuffer_size.value.y
			};
			return compbuffer_size;
		}
		StereoCompositorSideBySide( Vector<size_t,2> framebuffer_size ) :
			framebuffer_size( framebuffer_size ),
			compbuffer_size( getCompositedbufferSize( framebuffer_size ) )
		{
			compbuffer = (uint32_t*)malloc(sizeof(uint32_t) * compbuffer_size.value.x * compbuffer_size.value.y);
		}
		~StereoCompositorSideBySide()
		{
			free(compbuffer);
		}
		inline uint32_t* doCompositing(IceTImage* image)
		{
			static_assert(TController::pass_count >= 2, "Not enough passes defined in Controller for StereoCompositor!");
			uint32_t* left = icetImageGetColorui(image[0]);
			uint32_t* right = icetImageGetColorui(image[1]);
			for (unsigned int y = 0; y < compbuffer_size.value.y; y++)
			{
				memcpy( &(compbuffer[y*compbuffer_size.value.x                           ]), &( left[y*framebuffer_size.value.x]), sizeof(uint32_t) * framebuffer_size.value.x);
				memcpy( &(compbuffer[y*compbuffer_size.value.x + framebuffer_size.value.x]), &(right[y*framebuffer_size.value.x]), sizeof(uint32_t) * framebuffer_size.value.x);
			}
			return compbuffer;
		}
	private:
		Vector<size_t,2> compbuffer_size;
		Vector<size_t,2> framebuffer_size;
		uint32_t* compbuffer;
};

template <
	typename TController,
	uint32_t LeftFilter,
	uint32_t RightFilter
>
class StereoCompositorAnaglyph
{
	public:
		static inline Vector<size_t,2> getCompositedbufferSize( Vector<size_t,2> framebuffer_size )
		{
			return framebuffer_size;
		}
		StereoCompositorAnaglyph( Vector<size_t,2> framebuffer_size ) :
			framebuffer_size( framebuffer_size )
		{
			compbuffer = (uint32_t*)malloc(sizeof(uint32_t) * framebuffer_size.value.x * framebuffer_size.value.y);
		}
		~StereoCompositorAnaglyph()
		{
			free(compbuffer);
		}
		inline uint32_t* doCompositing(IceTImage* image)
		{
			static_assert(TController::pass_count >= 2, "Not enough passes defined in Controller for StereoCompositor!");
			uint32_t* left = icetImageGetColorui(image[0]);
			uint32_t* right = icetImageGetColorui(image[1]);
			for (unsigned int x = 0; x < framebuffer_size.value.x; x++)
				for (unsigned int y = 0; y < framebuffer_size.value.y; y++)
					compbuffer[x+y*framebuffer_size.value.x] =
						( left[x+y*framebuffer_size.value.x] &  LeftFilter) |
						(right[x+y*framebuffer_size.value.x] & RightFilter);
			return compbuffer;
		}
	private:
		Vector<size_t,2> framebuffer_size;
		uint32_t* compbuffer;
};

} //namespace isaac;
