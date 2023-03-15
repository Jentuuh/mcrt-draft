#pragma once
#include <glm/glm.hpp>

namespace mcrt {
	class GeometryUtils
	{
	public:
		GeometryUtils();

		static float triangleArea2D(glm::vec2 a, glm::vec2 b, glm::vec2 c);
		static float triangleArea3D(glm::vec3 a, glm::vec3 b, glm::vec3 c);
	};

}

