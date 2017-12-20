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

struct DefaultController
{
	static const int pass_count = 1;
	inline bool updateProjection( IceTDouble * const projection, const isaac_size2 & framebuffer_size, json_t * const message, const bool first = false)
	{
		if (first)
			setPerspective( projection, 45.0f, (isaac_float)framebuffer_size.x/(isaac_float)framebuffer_size.y,ISAAC_Z_NEAR, ISAAC_Z_FAR);
		return false;
	}
	inline void sendFeedback( json_t * const json_root, bool force = false ) {}
};

struct OrthoController
{
	static const int pass_count = 1;
	inline bool updateProjection( IceTDouble * const projection, const isaac_size2 & framebuffer_size, json_t * const message, const bool first = false)
	{
		if (first)
			setOrthographic( projection, (isaac_float)framebuffer_size.x/(isaac_float)framebuffer_size.y*isaac_float(2), isaac_float(2), ISAAC_Z_NEAR, ISAAC_Z_FAR);
		return false;
	}
	inline void sendFeedback( json_t * const json_root, bool force = false ) {}
};

struct StereoController
{
	static const int pass_count = 2;
	StereoController() :
		eye_distance(0.06f),
		send_stereo(true)
	{}
	inline bool updateProjection( IceTDouble * const projection, const isaac_size2 & framebuffer_size, json_t * const message, const bool first = false)
	{
		if ( json_t* js = json_object_get(message, "eye distance") )
		{
			send_stereo = true;
			eye_distance = json_number_value( js );
			json_object_del( message, "eye distance" );
		}
		if (first || send_stereo)
		{
			spSetPerspectiveStereoscopic( &(projection[ 0]), 45.0f, (isaac_float)framebuffer_size.x/(isaac_float)framebuffer_size.y,ISAAC_Z_NEAR, ISAAC_Z_FAR, 5.0f,  eye_distance);
			spSetPerspectiveStereoscopic( &(projection[16]), 45.0f, (isaac_float)framebuffer_size.x/(isaac_float)framebuffer_size.y,ISAAC_Z_NEAR, ISAAC_Z_FAR, 5.0f, -eye_distance);
		}
		return send_stereo;
	}
	inline void sendFeedback( json_t * const json_root, bool force = false )
	{
		if (send_stereo || force)
		{
			json_object_set_new( json_root, "eye distance", json_real( eye_distance ) );
			send_stereo = false;
		}
	}
	float eye_distance;
	bool send_stereo;
};

template <typename DrawController, typename FftSubController>
struct FftController
{
	static const int pass_count = DrawController::pass_count + FftSubController::pass_count;
	FftController() :
		drawController{},
		fftSubController{}
	{}
	inline bool updateProjection( IceTDouble * const projection, const isaac_size2 & framebuffer_size, json_t * const message, const bool first = false)
	{
		return
			drawController.updateProjection( projection, framebuffer_size, message, first ) ||
			fftSubController.updateProjection( projection + 16 * DrawController::pass_count, framebuffer_size, message, first );
	}
	inline void sendFeedback( json_t * const json_root, bool force = false )
	{
		drawController.sendFeedback( json_root, force );
		fftSubController.sendFeedback( json_root, force );
	}
	DrawController drawController;
	FftSubController fftSubController;
};

} //namespace isaac;
