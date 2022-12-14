#pragma once
#include "CUDABuffer.hpp"
#include "LaunchParams.hpp"
#include "camera.hpp"
#include "scene.hpp"
#include "default_pipeline.hpp"
#include "direct_light_pipeline.hpp"
#include "radiance_cell_gather_pipeline.hpp"
#include "radiance_cell_scatter_pipeline.hpp"

#include <stb/stb_image.h>

namespace mcrt {

	class Renderer {
	public:
		/*! Constructor : performs setup, including initializing OptiX, creation of module
		 pipelines, programs, SBT etc. */
		Renderer(Scene& scene, const Camera& camera);

		void render();

		void resize(const glm::ivec2& newSize);

		// Download rendered color buffer from device
		void downloadPixels(uint32_t h_pixels[]);
		void downloadDirectLighting(uint32_t h_pixels[]);
		void downloadAndWriteLightSourceTexture();

		// Update camera to render from
		void updateCamera(const Camera& camera);

	private:
		void writeToImage(std::string fileName, int resX, int resY, void* data);
		void initLightingTextures(int size);
		void calculateDirectLighting();
		void calculateIndirectLighting();
		void initSHWeightsBuffer(int amountNonEmptyCells);
		void initSHAccumulators(int divisionResolution, int amountNonEmptyCells);
		void calculateRadianceCellGatherPass(CUDABuffer& previousPassLightSourceTexture);
		void calculateRadianceCellScatterPass(int iteration, CUDABuffer& dstTexture);

		void loadLightTexture();
		void writeWeightsToTxtFile(std::vector<float>& weights, std::vector<int>& numSamples, int amountCells);

		void prepareUVWorldPositions();
		void prepareUVsInsideBuffer();

		// Helpers
		float area(glm::vec2 a, glm::vec2 b, glm::vec2 c);
		UVWorldData UVto3D(glm::vec2 uv);


	protected:
		// ------------------
		//	Internal helpers
		// ------------------

		// OptiX initialization, checking for errors
		void initOptix();

		// Creation + configuration of OptiX device context
		void createContext();

		// Fill geometryBuffers with scene geometry
		void fillGeometryBuffers();

		// Upload textures and create CUDA texture objects for them
		void createTextures();

	protected:
		// CUDA device context + stream that OptiX pipeline will run on,
		// and device properties of the device
		CUcontext			cudaContext;
		CUstream			stream;
		cudaDeviceProp		deviceProperties;

		// OptiX context that pipeline will run in
		OptixDeviceContext	optixContext;

		// Pipelines
		std::unique_ptr<DefaultPipeline> tutorialPipeline;
		std::unique_ptr<DirectLightPipeline> directLightPipeline;
		std::unique_ptr<RadianceCellGatherPipeline> radianceCellGatherPipeline;
		std::unique_ptr<RadianceCellScatterPipeline> radianceCellScatterPipeline;

		CUDABuffer lightSourceTexture; // UV map with direct light source (to test the SH projection)
		CUDABuffer colorBuffer;	// Framebuffer we will write to
		CUDABuffer directLightingTexture; // Texture in which we store the direct lighting
		CUDABuffer secondBounceTexture;	// Texture in which we store the second lighting bounce
		CUDABuffer thirdBounceTexture; // Texture in which we store the third lighting bounce
		CUDABuffer lightDataBuffer;	// In this buffer we'll store our light source data

		CUDABuffer nonEmptyCellDataBuffer;	// In this buffer we'll store our data for non empty radiance cells
		CUDABuffer SHWeightsDataBuffer; // In this buffer we'll store the SH weights
		CUDABuffer SHAccumulatorsBuffer; // In this buffer we'll store the SH accumulators
		CUDABuffer numSamplesAccumulatorsBuffer; // In this buffer we'll store the numSamples accumulators per radiance cell SH
		CUDABuffer UVWorldPositionDeviceBuffer; // In this buffer we'll store the world positions for each of our UV texels (starting from 0,0 --> 1,1), this means this array starts at the left bottom of the actual texture image
		CUDABuffer UVsInsideBuffer;		// In this buffer we'll store the UV worldpositions for all texels inside each cell
		CUDABuffer UVsInsideOffsets;	// In this buffer we'll store the offsets to index the UVsInsideBuffer

		Camera renderCamera;

		// Scene we are tracing rays against
		Scene& scene;

		// Device-side buffers (one buffer per input mesh!)
		std::vector<CUDABuffer> vertexBuffers;	
		std::vector<CUDABuffer> indexBuffers;
		std::vector<CUDABuffer> normalBuffers;
		std::vector<CUDABuffer> texcoordBuffers;
		std::vector<int> amountVertices;
		std::vector<int> amountIndices;

		std::vector<CUDABuffer> radianceGridVertexBuffers;
		std::vector<CUDABuffer> radianceGridIndexBuffers;
		std::vector<int> amountVerticesRadianceGrid;
		std::vector<int> amountIndicesRadianceGrid;


		std::vector<cudaArray_t>         textureArrays;
		std::vector<cudaTextureObject_t> textureObjects;
	};
}