#pragma once
#include "game_object.hpp"

// Geometry voxelizer to be used for proxy geometry and to test whether objects are in radiance cells.
namespace mcrt {
	struct Voxel {
		Voxel(glm::ivec3 coord, float cellSize, glm::vec3 minOffset)
		{
			min = glm::vec3{ minOffset.x + coord.x * cellSize, minOffset.y + coord.y * cellSize, minOffset.z + coord.z * cellSize };
			max = min + glm::vec3{ cellSize, cellSize, cellSize };
		}

		glm::vec3 min;
		glm::vec3 max;
	};

	class Voxelizer
	{
	public:
		Voxelizer(float voxelSize, std::shared_ptr<GameObject> object);

		void voxelize();

		std::vector<Voxel> resultVoxelGrid;
		std::shared_ptr<GameObject> voxelizedObject;
	private:
		void createBoundingBoxVoxelGrid();

		std::vector<Voxel> bbVoxelGrid;
		float voxelSize;
	};

}

