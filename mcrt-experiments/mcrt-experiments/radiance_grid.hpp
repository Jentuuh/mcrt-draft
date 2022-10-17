#pragma once
#include "radiance_cell.hpp"

#include "glm/glm.hpp"

#include <vector>

namespace mcrt {
	class RadianceGrid
	{
	public:
		RadianceGrid(float cellSize);

		RadianceCell& getCell(glm::ivec3 coord);

	private:
		glm::ivec3 resolution;
		std::vector<RadianceCell> grid;
	};
}


