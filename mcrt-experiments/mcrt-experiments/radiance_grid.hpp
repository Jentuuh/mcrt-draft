#pragma once
#include "radiance_cell.hpp"
#include "voxelizer.hpp"

#include "glm/glm.hpp"

#include <vector>

namespace mcrt {
	class RadianceGrid
	{
	public:
		RadianceGrid();
		void init(float cellSize);
		void assignObjectsToCells(std::vector<std::shared_ptr<Voxelizer>>& voxelizers);

		RadianceCell& getCell(glm::ivec3 coord);
		RadianceCell& getCell(int index) { return grid[index]; };
		std::vector<glm::vec3>& getVertices();
		std::vector<glm::ivec3>& getIndices();

		glm::ivec3 resolution;

	private:
		float cellSize = 1.0f;
		std::vector<RadianceCell> grid;
	};
}


