#include "radiance_grid.hpp"
#include "glm/gtx/string_cast.hpp"
#include <iostream>
namespace mcrt {

	RadianceGrid::RadianceGrid(){}

	void RadianceGrid::init(float cellSize)
	{
		// Assuming the scene is normalized and shifted into the positive quadrant
		resolution.x = 1.0f / cellSize;
		resolution.y = 1.0f / cellSize;
		resolution.z = 1.0f / cellSize;


		for (int z = 0; z < resolution.z; z++)
		{
			for (int y = 0; y < resolution.y; y++)
			{
				for (int x = 0; x < resolution.x; x++)
				{
					grid.push_back(RadianceCell{ glm::ivec3{x,y,z}, resolution, cellSize });
				}
			}
		}
	}


	RadianceCell& RadianceGrid::getCell(glm::ivec3 coord)
	{
		return grid[(coord.y * resolution.x) + (coord.z * resolution.x * resolution.y) + coord.x];
	}

	std::vector<glm::vec3>& RadianceGrid::getVertices()
	{
		std::vector<glm::vec3> allVertices;
		for (auto& r : grid)
		{
			allVertices.insert(allVertices.end(), r.getVertices().begin(), r.getVertices().end());
		}
		return allVertices;
	}


	std::vector<glm::ivec3>& RadianceGrid::getIndices()
	{
		std::vector<glm::ivec3> allIndices;
		for (auto& r : grid)
		{
			allIndices.insert(allIndices.end(), r.getIndices().begin(), r.getIndices().end());
		}
		return allIndices;
	}
}