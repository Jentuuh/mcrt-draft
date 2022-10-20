#pragma once

#include "CUDABuffer.hpp"
#include "LaunchParams.hpp"
#include "camera.hpp"
#include "scene.hpp"
#include "default_pipeline.hpp"

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

		// Update camera to render from
		void updateCamera(const Camera& camera);

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

		std::unique_ptr<DefaultPipeline> tutorialPipeline;

		CUDABuffer colorBuffer;	// Framebuffer we will write to

		Camera renderCamera;

		// Scene we are tracing rays against
		Scene& scene;

		// Device-side buffers (one buffer per input mesh!)
		std::vector<CUDABuffer> vertexBuffers;	
		std::vector<CUDABuffer> indexBuffers;
		std::vector<CUDABuffer> normalBuffers;
		std::vector<CUDABuffer> texcoordBuffers;

		std::vector<cudaArray_t>         textureArrays;
		std::vector<cudaTextureObject_t> textureObjects;
	};
}