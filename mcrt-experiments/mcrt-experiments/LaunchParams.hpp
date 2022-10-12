#pragma once

#include <glm/glm.hpp>
namespace mcrt {

	struct LaunchParams
	{
		int frameID{ 0 };
		uint32_t* colorBuffer;
		glm::uvec2 fbSize;
	};
}