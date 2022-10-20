#pragma once

#include "optix7.hpp"
#include "glm/glm.hpp"

namespace mcrt {

	enum { RADIANCE_RAY_TYPE = 0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };


	struct MeshSBTData {
		glm::vec3 color;
		glm::vec3 *vertex;
		glm::vec3* normal;
		glm::vec2* texcoord;
		glm::ivec3* index;

		bool hasTexture;
		cudaTextureObject_t texture;

		int objectType;
	};

	struct LaunchParams
	{
		int frameID = 0;

		struct {
			uint32_t* colorBuffer;
			glm::ivec2 size;
		} frame;

		struct {
			glm::vec3 position;
			glm::vec3 direction;
			glm::vec3 horizontal;
			glm::vec3 vertical;
		} camera;

		OptixTraversableHandle traversable;
	};
}