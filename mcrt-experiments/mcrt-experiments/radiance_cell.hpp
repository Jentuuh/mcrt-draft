#pragma once
#include "game_object.hpp"

#include <vector>

namespace mcrt {
	class RadianceCell
	{
	public:
		RadianceCell(glm::ivec3 coord, glm::ivec3 res, float scale);

		void addObject(std::shared_ptr<GameObject> obj);
		void removeObject(std::shared_ptr<GameObject> obj);

		std::vector<glm::vec3>& getVertices() { return vertices; };
		std::vector<glm::ivec3>& getIndices() { return indices; };

	private:
		std::vector<std::shared_ptr<GameObject>> objectsInside;

		std::vector<glm::vec3> vertices;
		std::vector<glm::ivec3> indices;
	};
}


