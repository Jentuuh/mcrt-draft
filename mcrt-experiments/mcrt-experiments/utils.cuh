#pragma once
#include <optix_device.h>
#include "LaunchParams.hpp"

using namespace mcrt;

namespace mcrt {
	struct ChildInfo {
		glm::vec3 min;
		glm::vec3 max;
		int index;
	};

	template<typename T>
	static __forceinline__ __device__ void swap(T* v1, T* v2)
	{
		T temp = *v1;
		*v1 = *v2;
		*v2 = temp;
	}

	// Finds projected point onto scene bounds, which we can use as a 'distant point' along a ray
	static __forceinline__ __device__ void find_distant_point_along_direction(glm::vec3 o, glm::vec3 dir, glm::vec3 cubeMin, glm::vec3 cubeMax, double* t_min, double* t_max)
	{
		*t_min = (cubeMin.x - o.x) / dir.x;
		*t_max = (cubeMax.x - o.x) / dir.x;


		double t_min_y = (cubeMin.y - o.y) / dir.y;
		double t_max_y = (cubeMax.y - o.y) / dir.y;

		if (*t_min > *t_max) swap(t_min, t_max);
		if (t_min_y > t_max_y) swap(&t_min_y, &t_max_y);

		if ((*t_min > t_max_y) || (t_min_y > *t_max)) {
			*t_min = NAN;
			*t_max = NAN;
			return;
		}

		if (t_min_y > *t_min)
			*t_min = t_min_y;

		if (t_max_y < *t_max)
			*t_max = t_max_y;

		double t_min_z = (cubeMin.z - o.z) / dir.z;
		double t_max_z = (cubeMax.z - o.z) / dir.z;

		if (t_min_z > t_max_z) swap(&t_min_z, &t_max_z);


		if ((*t_min > t_max_z) || (t_min_z > *t_max)) {
			*t_min = NAN;
			*t_max = NAN;
			return;
		}

		if (t_min_z > *t_min)
			*t_min = t_min_z;

		if (t_max_z < *t_max)
			*t_max = t_max_z;
		//printf("IN FUNCTION: minX %f minY %f minZ %f maxX %f maxY %f maxZ %f \n", *t_min, t_min_y, t_min_z, *t_max, t_max_y, t_max_z);
		//printf("IN FUNCTION: t-min %f t-max %f \n", *t_min,  *t_max);
	}

	static __forceinline__ __device__ bool isPointInBox(glm::vec3 min, glm::vec3 max, glm::vec3 point)
	{
		return ((point.x > min.x && point.x < max.x) && (point.y > min.y && point.y < max.y) && (point.z > min.z && point.z < max.z));
	}


	static __forceinline__ __device__ ChildInfo find_child(glm::vec3 min, glm::vec3 max, glm::vec3 vertexPosition)
	{
		ChildInfo result;

		glm::vec3 boxHalfSize = 0.5f * (max - min);
		glm::vec3 center = min + boxHalfSize;
		printf("Center: %f %f %f \n", center.x, center.y, center.z);

		// Index 0
		glm::vec3 min0 = center - glm::vec3{ boxHalfSize.x, 0.0f, boxHalfSize.z };
		glm::vec3 max0 = center + glm::vec3{ 0.0f, boxHalfSize.y, 0.0f };


		if (isPointInBox(min0, max0, vertexPosition))
		{
			printf("IN 0! \n");
			result.min = min0;
			result.max = max0;
			result.index = 0;
			return result;
		}

		// Index 1
		glm::vec3 min1 = center - glm::vec3{ 0.0f, 0.0f, boxHalfSize.z };
		glm::vec3 max1 = center + glm::vec3{ boxHalfSize.x, boxHalfSize.y, 0.0f };

		if (isPointInBox(min1, max1, vertexPosition))
		{
			printf("IN 1! \n");

			result.min = min1;
			result.max = max1;
			result.index = 1;
			return result;
		}

		// Index 2
		glm::vec3 min2 = min;
		glm::vec3 max2 = center;

		if (isPointInBox(min2, max2, vertexPosition))
		{
			printf("IN 2! \n");

			result.min = min2;
			result.max = max2;
			result.index = 2;
			return result;
		}

		// Index 3
		glm::vec3 min3 = center - glm::vec3{ 0.0f, boxHalfSize.y, boxHalfSize.z };
		glm::vec3 max3 = center + glm::vec3{ boxHalfSize.x, 0.0f, 0.0f };

		if (isPointInBox(min3, max3, vertexPosition))
		{
			printf("IN 3! \n");

			result.min = min3;
			result.max = max3;
			result.index = 3;
			return result;
		}

		// Index 4
		glm::vec3 min4 = center - glm::vec3{ boxHalfSize.x, 0.0f, 0.0f };
		glm::vec3 max4 = center + glm::vec3{ 0.0f, boxHalfSize.y, boxHalfSize.z };

		if (isPointInBox(min4, max4, vertexPosition))
		{
			printf("IN 4! \n");

			result.min = min4;
			result.max = max4;
			result.index = 4;
			return result;
		}

		// Index 5
		glm::vec3 min5 = center;
		glm::vec3 max5 = max;

		if (isPointInBox(min5, max5, vertexPosition))
		{
			printf("IN 5! \n");

			result.min = min5;
			result.max = max5;
			result.index = 5;
			return result;
		}

		// Index 6
		glm::vec3 min6 = center - glm::vec3{ boxHalfSize.x, boxHalfSize.y, 0.0f };
		glm::vec3 max6 = center + glm::vec3{ 0.0f, 0.0f, boxHalfSize.z }; ;

		if (isPointInBox(min6, max6, vertexPosition))
		{
			printf("IN 6! \n");

			result.min = min6;
			result.max = max6;
			result.index = 6;
			return result;
		}

		// Index 7
		glm::vec3 min7 = center - glm::vec3{ 0.0f, boxHalfSize.y, 0.0f };
		glm::vec3 max7 = center + glm::vec3{ boxHalfSize.x, 0.0f, boxHalfSize.z };

		if (isPointInBox(min7, max7, vertexPosition))
		{
			printf("IN 7! \n");

			result.min = min7;
			result.max = max7;
			result.index = 7;
			return result;
		}

		printf("AT END! \n");

	}
	static __forceinline__ __device__ glm::vec3 read_octree(glm::vec3 worldCoordinate, float* octree)
	{
		glm::vec3 pointerOrValue;
		bool isLeafNode = false;

		glm::vec3 min = glm::vec3{0.0f, 0.0f, 0.0f};
		glm::vec3 max = glm::vec3{1.0f, 1.0f, 1.0f};
		int parentOffset = 0;
		int depth = 0;
		while (!isLeafNode)
		{
			ChildInfo nextNode = find_child(min, max, worldCoordinate);
			//printf("Next index: %d\n", nextNode.index);
			//printf("Next min: %f %f %f\n", nextNode.min.x, nextNode.min.y, nextNode.min.z);
			//printf("Next max: %f %f %f\n", nextNode.max.x, nextNode.max.y, nextNode.max.z);

			pointerOrValue = glm::vec3{octree[parentOffset + nextNode.index * 4 + 0], octree[parentOffset + nextNode.index * 4 + 1], octree[parentOffset + nextNode.index * 4 + 2] };
			isLeafNode = octree[parentOffset + nextNode.index * 4 + 3] != 0.5f;	// If the node is not a pointer type, it is a leaf node (either empty or containing a value)
			
			//printf("Node value: %f , %f , %f, %f \n", pointerOrValue.x, pointerOrValue.y, pointerOrValue.z, octree[parentOffset + nextNode.index * 4 + 3]);

			min = nextNode.min;
			max = nextNode.max;
			parentOffset = pointerOrValue.z * 1024 * 1024 + pointerOrValue.y * 1024 + pointerOrValue.x * 32;
			depth++;
		}

		printf("Octree value found! Depth traversed: %d\n", depth);
		return pointerOrValue;
	}

	static __forceinline__ __device__ void write_octree(glm::vec3 worldCoordinate, glm::vec3 value, float* octree)
	{
		glm::vec3 pointerOrValue;
		bool isLeafNode = false;

		glm::vec3 min = glm::vec3{ 0.0f, 0.0f, 0.0f };
		glm::vec3 max = glm::vec3{ 1.0f, 1.0f, 1.0f };
		int parentOffset = 0;
		int depth = 0;
		while (!isLeafNode)
		{
			ChildInfo nextNode = find_child(min, max, worldCoordinate);
			printf("Next index: %d\n", nextNode.index);
			printf("Next min: %f %f %f\n", nextNode.min.x, nextNode.min.y, nextNode.min.z);
			printf("Next max: %f %f %f\n", nextNode.max.x, nextNode.max.y, nextNode.max.z);

			pointerOrValue = glm::vec3{ octree[parentOffset + nextNode.index * 4 + 0], octree[parentOffset + nextNode.index * 4 + 1], octree[parentOffset + nextNode.index * 4 + 2] };
			isLeafNode = octree[parentOffset + nextNode.index * 4 + 3] != 0.5f;	// If the node is not a pointer type, it is a leaf node (either empty or containing a value)

			if (isLeafNode)
			{
				octree[parentOffset + nextNode.index * 4 + 0] = value.x;
				octree[parentOffset + nextNode.index * 4 + 1] = value.y;
				octree[parentOffset + nextNode.index * 4 + 2] = value.z;
				octree[parentOffset + nextNode.index * 4 + 3] = 1.0f;
			}

			min = nextNode.min;
			max = nextNode.max;
			parentOffset = pointerOrValue.z * 1024 * 1024 + pointerOrValue.y * 1024 + pointerOrValue.x * 32;
			depth++;
		}

		printf("Wrote value to octree! Depth traversed: %d\n", depth);
	}

}