#include "geometry_utils.hpp"

namespace mcrt {
	GeometryUtils::GeometryUtils(){}

	float GeometryUtils::triangleArea2D(glm::vec2 a, glm::vec2 b, glm::vec2 c)
	{
		// cross product / 2
		glm::vec2 v1 = a - c;
		glm::vec2 v2 = b - c;
		return (v1.x * v2.y - v1.y * v2.x) / 2.0f;
	}

	float GeometryUtils::triangleArea3D(glm::vec3 a, glm::vec3 b, glm::vec3 c)
	{
		// length of cross product (area of parallellogram) divided by 2
		return glm::length(glm::cross(b - a, c - a)) / 2.0f;
	}
}