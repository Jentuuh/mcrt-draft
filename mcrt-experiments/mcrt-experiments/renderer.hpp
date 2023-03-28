#pragma once
#include "CUDABuffer.hpp"
#include "LaunchParams.hpp"
#include "camera.hpp"
#include "image.hpp"
#include "octree_texture.hpp"
#include "scene.hpp"
#include "helpers.hpp"
#include "default_pipeline.hpp"
#include "default_pipeline_octree.hpp"
#include "direct_light_pipeline.hpp"
#include "direct_light_pipeline_octree.hpp"
#include "radiance_cell_gather_pipeline.hpp"
#include "radiance_cell_gather_cube_map_pipeline.hpp"
#include "radiance_cell_gather_cube_map_pipeline_octree.hpp"
#include "radiance_cell_scatter_pipeline.hpp"
#include "radiance_cell_scatter_cube_map_pipeline.hpp"
#include "radiance_cell_scatter_cube_map_pipeline_octree.hpp"
#include "radiance_cell_scatter_unbiased_pipeline.hpp"
#include "radiance_cell_scatter_unbiased_pipeline_octree.hpp"
#include "geometry_utils.hpp"
#include "general_utils.hpp"
#include "hdr_image.hpp"

#include <stb/stb_image.h>

namespace mcrt {
	
	enum BIAS_MODE {
		UNBIASED,
		BIASED_PROBES
	};

	enum WORLD_DATA_LOAD {
		LOAD_WORLD_DATA,
		CALCULATE_WORLD_DATA
	};

	enum PROBE_MODE {
		CUBE_MAP,
		SPHERICAL_HARMONICS,
		NA
	};

	enum IRRADIANCE_STORAGE_TYPE {
		TEXTURE_2D,
		OCTREE_TEXTURE
	};

	class Renderer {
	public:
		/*! Constructor : performs setup, including initializing OptiX, creation of module
		 pipelines, programs, SBT etc. */
		Renderer(Scene& scene, const Camera& camera, BIAS_MODE bias, PROBE_MODE probeType, IRRADIANCE_STORAGE_TYPE irradianceStorage);

		void render();

		void resize(const glm::ivec2& newSize);

		// Download rendered color buffer from device
		void downloadPixels(uint32_t h_pixels[]);

		// Update camera to render from
		void updateCamera(const Camera& camera);

	private:
		void initLightingTexturesPerObject();
		void initLightProbeCubeMaps(int resolution, int gridRes);

		// 2D texture based - Cubemap and SH
		void calculateDirectLighting();
		void calculateIndirectLighting(BIAS_MODE bias, PROBE_MODE mode);
		void initSHWeightsBuffer(int amountNonEmptyCells);
		void initSHAccumulators(int divisionResolution, int amountNonEmptyCells);
		void calculateRadianceCellGatherPass(CUDABuffer& previousPassLightSourceTexture);
		void calculateRadianceCellGatherPassCubeMapAlt(cudaTextureObject_t* previousPassLightSourceTextures);
		void calculateRadianceCellScatterPass(int iteration, CUDABuffer& dstTexture);
		void calculateRadianceCellScatterPassCubeMap(int iteration, cudaTextureObject_t* prevBounceTexture, cudaSurfaceObject_t* dstTexture);
		void calculateRadianceCellScatterUnbiased(int iteration, cudaTextureObject_t* prevBounceTexture, cudaSurfaceObject_t* dstTexture);

		// Octree
		void calculateDirectLightingOctree();
		void calculateIndirectLightingOctree(BIAS_MODE bias, PROBE_MODE mode);
		void calculateRadianceCellGatherPassCubeMapAltOctree(CUDABuffer& previousPassLightSourceOctreeTexture);
		void calculateRadianceCellScatterPassCubeMapOctree(int iteration, CUDABuffer& prevBounceOctreeTexture, CUDABuffer& dstOctreeTexture);
		void calculateRadianceCellScatterUnbiasedOctree(int iteration, CUDABuffer& prevBounceOctreeTexture, CUDABuffer& dstOctreeTexture);

		// Generate world sample data
		void prepareUVWorldPositionsPerObject(WORLD_DATA_LOAD loadOrCalculate);
		void prepareUVsInsideBuffer();
		void prepareWorldSamplePoints(float octreeLeafFaceArea);

		// Tests
		void lightProbeTest(int iteration, CUDABuffer& prevBounceTexture, CUDABuffer& dstTexture);
		void octreeTextureTest();
		void textureAndSurfaceObjectTest();

		// Helpers
		void barycentricCoordinates(glm::vec3 p, glm::vec3 a, glm::vec3 b, glm::vec3 c, float& u, float& v, float& w);
		UVWorldData UVto3D(glm::vec2 uv);
		UVWorldData UVto3DPerObject(glm::vec2 uv, std::shared_ptr<GameObject> o);
		void writeWorldDataTextureToFile(std::vector<float>& worldDataTexture, std::string fileName);
		void readWorldDataTextureFromFile(std::string fileName, std::vector<float>& container);

		// Debug
		void writeUVsPerCellToImage(std::vector<int>& offsets, std::vector<glm::vec2>& uvs, int texRes);
		void loadLightTexture();
		void writeWeightsToTxtFile(std::vector<float>& weights, std::vector<int>& numSamples, int amountCells);

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
		void createDiffuseTextures();

		// Upload textures and create CUDA texture objects for them
		void createWorldDataTextures();

	protected:
		// Renderer properties
		IRRADIANCE_STORAGE_TYPE irradStorageType;

		// CUDA device context + stream that OptiX pipeline will run on,
		// and device properties of the device
		CUcontext			cudaContext;
		CUstream			stream;
		cudaDeviceProp		deviceProperties;

		// OptiX context that pipeline will run in
		OptixDeviceContext	optixContext;

		// 2D texture-based pipelines
		std::unique_ptr<DefaultPipeline> cameraPipeline;
		std::unique_ptr<DirectLightPipeline> directLightPipeline;
		std::unique_ptr<RadianceCellGatherPipeline> radianceCellGatherPipeline;
		std::unique_ptr<RadianceCellGatherCubeMapPipeline> radianceCellGatherCubeMapPipeline;
		std::unique_ptr<RadianceCellScatterPipeline> radianceCellScatterPipeline;
		std::unique_ptr<RadianceCellScatterCubeMapPipeline> radianceCellScatterCubeMapPipeline;
		std::unique_ptr<RadianceCellScatterUnbiasedPipeline> radianceCellScatterUnbiasedPipeline;

		// Octree texture-based pipelines
		std::unique_ptr<DefaultPipelineOctree> cameraPipelineOctree;
		std::unique_ptr<DirectLightPipelineOctree> directLightPipelineOctree;
		std::unique_ptr<RadianceCellGatherCubeMapPipelineOctree> radianceCellGatherCubeMapPipelineOctree;
		std::unique_ptr<RadianceCellScatterCubeMapPipelineOctree> radianceCellScatterCubeMapPipelineOctree;
		std::unique_ptr<RadianceCellScatterUnbiasedPipelineOctree> radianceCellScatterUnbiasedPipelineOctree;

		CUDABuffer lightSourceTexture; // UV map with direct light source (to test the SH projection)
		CUDABuffer colorBuffer;	// Framebuffer we will write to


		CUDABuffer lightDataBuffer;	// In this buffer we'll store our light source data
		CUDABuffer cubeMaps; // In this buffer we'll store the light probe cubemaps

		// TODO: clean up!
		CUDABuffer nonEmptyCellDataBuffer;	// In this buffer we'll store our data for non empty radiance cells
		CUDABuffer SHWeightsDataBuffer; // In this buffer we'll store the SH weights
		CUDABuffer SHAccumulatorsBuffer; // In this buffer we'll store the SH accumulators
		CUDABuffer numSamplesAccumulatorsBuffer; // In this buffer we'll store the numSamples accumulators per radiance cell SH

		CUDABuffer samplePointWorldPositionDeviceBuffer; // In this buffer we'll store the world positions for each of our UV texels (starting from 0,0 --> 1,1), this means this array starts at the left bottom of the actual texture image
		CUDABuffer UVsInsideBuffer;		// In this buffer we'll store the UV worldpositions for all texels inside each cell
		CUDABuffer UVsInsideOffsets;	// In this buffer we'll store the offsets to index the UVsInsideBuffer
		CUDABuffer UVsGameObjectNrsBuffer; // In this buffer we'll store the game object identifiers that show for each UV to which game object it belongs (so we know from which texture we should read)

		std::unique_ptr<OctreeTexture> octreeTextures;

		Camera renderCamera;

		// Scene we are tracing rays against
		Scene& scene;

		// Device-side buffers (one buffer per input mesh!)
		std::vector<CUDABuffer> vertexBuffers;	
		std::vector<CUDABuffer> indexBuffers;
		std::vector<CUDABuffer> normalBuffers;
		std::vector<CUDABuffer> texcoordBuffers;
		std::vector<CUDABuffer> diffuseTextureUVBuffers;

		std::vector<int> amountVertices;
		std::vector<int> amountIndices;

		std::vector<CUDABuffer> radianceGridVertexBuffers;
		std::vector<CUDABuffer> radianceGridIndexBuffers;
		std::vector<int> amountVerticesRadianceGrid;
		std::vector<int> amountIndicesRadianceGrid;

		std::vector<int>				 directTextureSizes;
		std::vector<cudaArray_t>         textureArrays;
		std::vector<cudaTextureObject_t> diffuseTextureObjects;
		CUDABuffer diffuseTextureObjectPointersBuffer;

		std::vector<cudaTextureObject_t> textureObjectsDirect;
		std::vector<cudaSurfaceObject_t> surfaceObjectsDirect;
		std::vector<cudaTextureObject_t> textureObjectsSecond;
		std::vector<cudaSurfaceObject_t> surfaceObjectsSecond;
		std::vector<cudaTextureObject_t> textureObjectsThird;
		std::vector<cudaSurfaceObject_t> surfaceObjectsThird;

		CUDABuffer directTextureObjectPointersBuffer;
		CUDABuffer secondBounceTextureObjectPointersBuffer;
		CUDABuffer thirdBounceTextureObjectPointersBuffer;

		CUDABuffer directSurfaceObjectPointersBuffer;
		CUDABuffer secondSurfaceObjectPointersBuffer;
		CUDABuffer thirdSurfaceObjectPointersBuffer;
		CUDABuffer textureSizesBuffer;

		std::vector<cudaTextureObject_t> UVWorldPositionsTextures;
		std::vector<cudaTextureObject_t> UVNormalsTextures;
		std::vector<cudaTextureObject_t> UVDiffuseColorTextures;
		std::vector<cudaTextureObject_t> diffuseTextureUVsTextures; // Textures that contain the diffuse texture UVs for each texel of an object (since we need both lightmapped UVs and diff. texture UVs)
		std::vector<int> objectHasTexture;
		CUDABuffer uvWorldPositionTextureObjectPointersBuffer;
		CUDABuffer uvNormalTextureObjectPointersBuffer;
		CUDABuffer uvDiffuseColorTextureObjectPointersBuffer;
		CUDABuffer diffuseTextureUVsTextureObjectPointersBuffer;
		CUDABuffer objectHasTextureBuffer;
	};
}