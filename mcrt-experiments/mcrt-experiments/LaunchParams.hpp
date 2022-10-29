#pragma once

#include "optix7.hpp"
#include "glm/glm.hpp"
#include "area_light.hpp"
#include <vector>

namespace mcrt {

	enum { RADIANCE_RAY_TYPE = 0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };

	struct MeshSBTDataDirectLighting {
		glm::vec3* vertex;
		glm::vec3* normal;
		glm::vec2* texcoord;
		glm::ivec3* index;
	};


	struct LaunchParamsDirectLighting {
		struct {
			glm::vec3* positionsBuffer;
			int size;
		} uvWorldPositions;

		struct {
			uint32_t* colorBuffer;
			int size;
		} directLightingTexture;

		LightData* lights;
		int amountLights;
		int stratifyResX;
		int stratifyResY;

		OptixTraversableHandle traversable;
	};


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


	struct LaunchParamsTutorial
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