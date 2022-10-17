#pragma once

#include "glm/glm.hpp"
#include <vector>

namespace mcrt {
	class Model
	{
	public:
		Model();

		void loadModel();

		std::vector<glm::vec3> vertices;
		std::vector<glm::ivec3> indices;
	};
}

