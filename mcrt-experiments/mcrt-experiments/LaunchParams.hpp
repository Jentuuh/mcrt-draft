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
		float* colorBuffer;
		int size;
	};


	struct SHWeights {
		float* weights;
		int size;
		int amountBasisFunctions;
	};

	struct UVWorldData {
		glm::vec3 worldPosition;
		glm::vec3 worldNormal;
		glm::vec3 diffuseColor;
	};

	/**
	* ==================================
	*	 RADIANCE CELL SCATTER PASS
	* ==================================
	*/
	struct RadianceCellScatterPRD {
		float distanceToClosestIntersectionSquared;
		glm::vec3 rayOrigin;
	};

	struct RadianceCellScatterPRDHybrid {
		int hitFound;
		glm::vec3 resultColor;
	};

	struct MeshSBTDataRadianceCellScatter{
		glm::vec3* vertex;
		glm::vec3* normal;
		glm::vec2* texcoord;
		glm::ivec3* index;

		int cellIndex;
	};


	struct LaunchParamsRadianceCellScatter {
		struct {
			UVWorldData* UVDataBuffer;
			int size;
		} uvWorldPositions;

		SHWeights sphericalHarmonicsWeights;

		PixelBuffer currentBounceTexture;

		glm::vec2* uvsInside;
		int* uvsInsideOffsets;
		glm::vec3 cellCenter;
		float cellSize;

		int stratifyResX;
		int stratifyResY;

		int nonEmptyCellIndex;

		OptixTraversableHandle sceneTraversable;
	};

	struct LaunchParamsRadianceCellScatterCubeMap
	{
		struct {
			UVWorldData* UVDataBuffer;
			int size;
		} uvWorldPositions;

		PixelBuffer prevBounceTexture;
		PixelBuffer currentBounceTexture;

		glm::vec2* uvsInside;
		int* uvsInsideOffsets;
		glm::vec3 cellCenter;
		float cellSize;

		glm::ivec3 cellCoords;
		int probeWidthRes;	// Amount of probes in the x-direction
		int probeHeightRes;	// Amount of probes in the y-direction
		int nonEmptyCellIndex;

		float* cubeMaps; // A pointer to cubemap faces
		int cubeMapResolution;

		OptixTraversableHandle sceneTraversable;
	};


	struct RadianceCellScatterUnbiasedPRD {
		glm::vec3 resultColor;
	};

	struct LaunchParamsRadianceCellScatterUnbiased
	{
		struct {
			UVWorldData* UVDataBuffer;
			int size;
		} uvWorldPositions;

		PixelBuffer prevBounceTexture;
		PixelBuffer currentBounceTexture;

		glm::vec2* uvsInside;
		int* uvsInsideOffsets;
		glm::vec3 cellCenter;
		float cellSize;
		int nonEmptyCellIndex;
		int lightSrcU;
		int lightSrcV;

		OptixTraversableHandle sceneTraversable;
	};



	/**
	* ==================================
	*	 RADIANCE CELL GATHER PASS
	* ==================================
	*/

	struct RadianceCellGatherPRD {
		float distanceToClosestProxyIntersectionSquared;
		glm::vec3 rayOrigin;
	};

	struct RadianceCellGatherPRDAlt {
		glm::vec3 resultColor;
	};

	struct MeshSBTDataRadianceCellGather {
		glm::vec3* vertex;
		glm::vec3* normal;
		glm::ivec3* index;
		glm::vec2* texcoord;

		int cellIndex;
	};

	struct LaunchParamsRadianceCellGatherCubeMap {
		struct {
			UVWorldData* UVDataBuffer;
			int size;
		} uvWorldPositions;

		//struct {
		//	glm::vec3* centers;
		//	int size;
		//} nonEmptyCells;

		float cellSize;
		int divisionResolution;		// The amount of cells the light source texture is divided in both dimensions

		glm::vec3 probePosition;
		int probeOffset;

		PixelBuffer lightSourceTexture;

		float* cubeMaps; // A pointer to cubemap faces
		int cubeMapResolution;

		OptixTraversableHandle sceneTraversable;
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
		int divisionResolution;		// The amount of cells the light source texture is divided in both dimensions

		int nonEmptyCellIndex;

		float* shAccumulators;
		int* shNumSamplesAccumulators;
		SHWeights sphericalHarmonicsWeights; // Radiance cells thus need to get an index in their SBT data so we can index the weights array by that index!
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
		glm::vec3 diffuseColor;
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

		PixelBuffer lightTexture;
		PixelBuffer lightTextureSecondBounce;
		PixelBuffer lightTextureThirdBounce;

		OptixTraversableHandle traversable;
	};
}