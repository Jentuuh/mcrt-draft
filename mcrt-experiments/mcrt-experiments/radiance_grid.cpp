#include "radiance_grid.hpp"


namespace mcrt {

	RadianceGrid::RadianceGrid(float cellSize)
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
					grid.push_back(RadianceCell{ glm::ivec3{x,y,z}, cellSize });
				}
			}
		}
	}


	RadianceCell& RadianceGrid::getCell(glm::ivec3 coord)
	{
		return grid[(coord.y * resolution.x) + (coord.z * resolution.x * resolution.y) + coord.x];
	}

}