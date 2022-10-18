#pragma once

#include "glm/glm.hpp"
#include <string>
#include <vector>
#include <memory>


namespace mcrt {
	struct TriangleMesh {
		std::vector<glm::vec3> vertices;
		std::vector<glm::ivec3> indices;
		std::vector<glm::vec2> texCoords;
		std::vector<glm::vec3> normals;

		// material data
		glm::vec3 diffuse;
	};

	class Model
	{
	public:
		Model();

		void loadModel();

		std::shared_ptr<TriangleMesh> mesh;
	};
}

