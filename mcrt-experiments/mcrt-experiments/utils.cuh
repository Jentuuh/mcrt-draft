#pragma once
#include <optix_device.h>
#include "LaunchParams.hpp"

using namespace mcrt;

namespace mcrt {
	
	//// Finds projected point onto scene bounds, which we can use as a 'distant point' along a ray
	//static __forceinline__ __device__ void find_distant_point_along_direction(glm::vec3 o, glm::vec3 dir, glm::vec3* dst)
	//{
	//	float maxExtent = max(max(abs(dir.x), abs(dir.y)), abs(dir.z));
	//	glm::vec3 projection;

	//	// X principal axis
	//	if (maxExtent == abs(dir.x))
	//	{
	//		if (dir.x > 0) // positive --> round to 1
	//		{
	//			projection.x = o.x + 1.0f;
	//			float t = (projection.x - o.x) / dir.x;	// positive divided by positive is positive
	//			projection = o + t * dir;
	//		}
	//		else { // negative --> round to 0
	//			projection.x = o.x - 1.0f;
	//			float t = (projection.x - o.x) / dir.x; // negative divided by negative becomes positive
	//			projection = o + t * dir;
	//		}
	//	}
	//	// Y principal axis
	//	else if (maxExtent == abs(dir.y))
	//	{
	//		if (dir.y > 0) // positive --> round to 1
	//		{
	//			projection.y = o.y + 1.0f;
	//			float t = (projection.y - o.y) / dir.y;	// positive divided by positive is positive
	//			projection = o + t * dir;
	//		}
	//		else { // negative --> round to 0
	//			projection.y = o.y - 1.0f;
	//			float t = (projection.y - o.y) / dir.y; // negative divided by negative becomes positive
	//			projection = o + t * dir;
	//		}
	//	}
	//	// Z principal axis
	//	else if (maxExtent == abs(dir.z))
	//	{
	//		if (dir.z > 0) // positive --> round to 1
	//		{
	//			projection.z = o.z + 1.0f;
	//			float t = (projection.z - o.z) / dir.z;	// positive divided by positive is positive
	//			projection = o + t * dir;

	//		}
	//		else { // negative --> round to 0
	//			projection.z = o.z - 1.0f;
	//			float t = (projection.z - o.z) / dir.z; // negative divided by negative becomes positive
	//			projection = o + t * dir;
	//		}
	//	}

	//	*dst = projection;;
	//}

	// Finds projected point onto scene bounds, which we can use as a 'distant point' along a ray
	static __forceinline__ __device__ void find_distant_point_along_direction(glm::vec3 o, glm::vec3 dir, glm::vec3* dst)
	{
		double max_projections[3] = {
			fabs(dir.x),
			fabs(dir.y),
			fabs(dir.z)
		};

		int max_index = 0;
		for (int i = 1; i < 3; i++) {
			if (max_projections[i] > max_projections[max_index]) {
				max_index = i;
			}
		}

		switch (max_index) {
		case 0:
			dst->x = dir.x > 0 ? 1 : 0;
			dst->y = o.y + (dst->x - o.x) * dir.y / dir.x;
			dst->z = o.z + (dst->x - o.x) * dir.z / dir.x;
			break;
		case 1:
			dst->y = dir.y > 0 ? 1 : 0;
			dst->x = o.x + (dst->y - o.y) * dir.x / dir.y;
			dst->z = o.z + (dst->y - o.y) * dir.z / dir.y;
			break;
		case 2:
			dst->z = dir.z > 0 ? 1 : 0;
			dst->x = o.x + (dst->z - o.z) * dir.x / dir.z;
			dst->y = o.y + (dst->z - o.z) * dir.y / dir.z;
			break;
		}

		// Clamp the result to the boundaries of the unit cube
		dst->x = clamp(dst->x, 0.0, 1.0);
		dst->y = clamp(dst->y, 0.0, 1.0);
		dst->z = clamp(dst->z, 0.0, 1.0);
	}
}