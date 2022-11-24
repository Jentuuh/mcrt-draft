#pragma once

#include "optix7.hpp"
#include "glm/glm.hpp"
#include "area_light.hpp"
#include <vector>

namespace mcrt {

	enum { RADIANCE_RAY_TYPE = 0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };

	/**
	* ================
	*	HELPER TYPES 
	* ================
	*/
	struct PixelBuffer {
		uint32_t* colorBuffer;
		int size;
	};


	struct SHWeights {
		double* weights;
		int size;
		int amountBasisFunctions;
	};

	struct UVWorldData {
		glm::vec3 worldPosition;
		glm::vec3 worldNormal;
	};


	/**
	* ==================================
	*	 RADIANCE CELL GATHER PASS
	* ==================================
	*/

	struct RadianceCellGatherPRD {
		float distanceToClosestProxyIntersection;
		glm::vec3 rayOrigin;
	};

	struct MeshSBTDataRadianceCellGather {
		glm::vec3* vertex;
		glm::vec3* normal;
		glm::ivec3* index;

		int cellIndex;
	};

	struct LaunchParamsRadianceCellGather {
		struct {
			UVWorldData* UVDataBuffer;
			int size;
		} uvWorldPositions;

		struct {
			glm::vec3* centers;
			int size;
		} nonEmptyCells;

		float cellSize;
		int stratifyResX;
		int stratifyResY;

		SHWeights sphericalHarmonicsWeights;	// Radiance cells thus need to get an index in their SBT data so we can index the weights array by that index!
		PixelBuffer lightSourceTexture;

		OptixTraversableHandle sceneTraversable;
		//OptixTraversableHandle gasTraversables[2];
		//OptixTraversableHandle iasTraversable;
	};


	/**
	* ==============================
	*	  DIRECT LIGHTING PASS
	* ==============================
	*/

	struct DirectLightingPRD {
		glm::vec3 lightSamplePos;
		glm::vec3 rayOrigin;
		glm::vec3 resultColor;
	};

	struct MeshSBTDataDirectLighting {
		glm::vec3* vertex;
		glm::vec3* normal;
		glm::vec2* texcoord;
		glm::ivec3* index;
	};


	struct LaunchParamsDirectLighting {
		struct {
			UVWorldData* UVDataBuffer;
			int size;
		} uvWorldPositions;

		PixelBuffer directLightingTexture;

		LightData* lights;
		int amountLights;
		int stratifyResX;
		int stratifyResY;

		OptixTraversableHandle traversable;
	};


	/**
	* =======================
	*	  TUTORIAL PASS
	* =======================
	*/

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