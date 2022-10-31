#include "radiance_grid.hpp"
#include "glm/gtx/string_cast.hpp"
#include <iostream>
namespace mcrt {

	RadianceGrid::RadianceGrid(){}

	void RadianceGrid::init(float cellSize)
	{
		this->cellSize = cellSize;
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

	void RadianceGrid::assignObjectsToCells(std::vector<std::shared_ptr<GameObject>>& objects)
	{
		for (auto& g : objects)
		{
			AABB objectWorldAABB = g->getWorldAABB();
			std::cout << glm::to_string(objectWorldAABB.min) << glm::to_string(objectWorldAABB.max) << std::endl;
			// floor(minCoords / cellSize)
			glm::ivec3 minRadianceCoords = { std::floorf(objectWorldAABB.min.x / cellSize), std::floorf(objectWorldAABB.min.y / cellSize), std::floorf(objectWorldAABB.min.z / cellSize) };
			glm::ivec3 maxRadianceCoords = { std::floorf(objectWorldAABB.max.x / cellSize), std::floorf(objectWorldAABB.max.y / cellSize), std::floorf(objectWorldAABB.max.z / cellSize) };
			std::cout << glm::to_string(minRadianceCoords) << glm::to_string(maxRadianceCoords) << std::endl;
			//for (int x = minRadianceCoords.x; x < maxRadianceCoords.x; x++)
			//{
			//	for (int y = minRadianceCoords.y; y < maxRadianceCoords.y; y++)
			//	{
			//		for (int z = minRadianceCoords.z; z < maxRadianceCoords.z; z++)
			//		{
			//			RadianceCell& currentCell = getCell({ x,y,z });
			//			for (auto& v : g->getWorldVertices())
			//			{
			//				// If the cell contains a vertex we can add the game object to this cell
			//				if (currentCell.contains(v)) {
			//					std::cout << "Youpie!" << std::endl;
			//					currentCell.addObject(g);
			//					break;
			//				}
			//			}
			//		}
			//	}
			//}
			for (auto& c : grid)
			{
				for (auto& v : g->getWorldVertices())
				{
					// If the cell contains a vertex we can add the game object to this cell
					if (c.contains(v)) {
						std::cout << "Youpie!" << std::endl;
						c.addObject(g);
						break;
					}
				}
			}
		}

		// Print amount of objects for each cell
		int idx = 0;
		for (auto& c : grid)
		{
			std::cout << "Cell " << idx << ": " << c.amountObjects() << " objects." << std::endl;
			idx++;
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