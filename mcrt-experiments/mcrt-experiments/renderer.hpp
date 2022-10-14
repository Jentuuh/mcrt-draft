#pragma once

#include "CUDABuffer.hpp"
#include "LaunchParams.hpp"

namespace mcrt {

	struct Camera {
		glm::vec3 eye;
		glm::vec3 target;
		glm::vec3 up;
	};

	// Simple indexed triangle mesh container
	struct TriangleMesh {

		// Add unit cube, subject to given cfm matrix, to the triangle mesh
		void addUnitCube(const glm::mat4x4& xfm);
		// Add aligned cube given its front-lower-left corner + size
		void addCube(const glm::vec3& center, const glm::vec3& size);

		std::vector<glm::vec3> vertex;
		std::vector<glm::ivec3> index;
	};

	class Renderer {
	public:
		/*! Constructor : performs setup, including initializing OptiX, creation of module
		 pipelines, programs, SBT etc. */
		Renderer(const TriangleMesh& model);

		void render();

		void resize(const glm::ivec2& newSize);

		// Download rendered color buffer from device
		void downloadPixels(uint32_t h_pixels[]);

		// Set camera to render from
		void setCamera(const Camera& camera);

	protected:
		// ------------------
		//	Internal helpers
		// ------------------

		// OptiX initialization, checking for errors
		void initOptix();

		// Creation + configuration of OptiX device context
		void createContext();

		/*! creates the module that contains all the programs we are going
		  to use. in this simple example, we use a single module from a
		  single .cu file, using a single embedded ptx string */
		void createModule();

		// Setup for raygen programs(s) that we'll use
		void createRaygenPrograms();

		// Setup for miss programs(s) that we'll use
		void createMissPrograms();

		// Setup for hitgroup programs(s) that we'll use
		void createHitGroupPrograms();

		// Assembles full pipeline of all programs
		void createPipeline();

		// Construction of shader binding table
		void buildSBT();

		// Build acceleration structure for our mesh
		OptixTraversableHandle buildAccel(const TriangleMesh& model);

	protected:
		// CUDA device context + stream that OptiX pipeline will run on,
		// and device properties of the device
		CUcontext			cudaContext;
		CUstream			stream;
		cudaDeviceProp		deviceProperties;

		// OptiX context that pipeline will run in
		OptixDeviceContext	optixContext;

		// Pipeline 
		OptixPipeline				pipeline;
		OptixPipelineCompileOptions	pipelineCompileOptions = {};
		OptixPipelineLinkOptions	pipelineLinkOptions = {};

		// Module containing device programs
		OptixModule					module;
		OptixModuleCompileOptions	moduleCompileOptions = {};

		// Vectors of all program(group)s, and SBT built around them
		std::vector<OptixProgramGroup> raygenPGs;
		CUDABuffer raygenRecordsBuffer;
		std::vector<OptixProgramGroup> missPGs;
		CUDABuffer missRecordsBuffer;
		std::vector<OptixProgramGroup> hitgroupPGs;
		CUDABuffer hitgroupRecordsBuffer;
		OptixShaderBindingTable sbt = {};

		// Launch params (accessible from each program) on host, 
		// and buffer to store them on device
		LaunchParams launchParams;
		CUDABuffer   launchParamsBuffer;

		CUDABuffer colorBuffer;	// Framebuffer we will write to

		Camera renderCamera;

		// Model we are tracing rays against
		const TriangleMesh model;
		// Device-side buffers
		CUDABuffer vertexBuffer;	
		CUDABuffer indexBuffer;
		CUDABuffer accelerationStructBuffer;
	};
}