#pragma once

#include "CUDABuffer.hpp"
#include "LaunchParams.hpp"

namespace mcrt {

	class Renderer {
	public:
		/*! Constructor : performs setup, including initializing OptiX, creation of module
		 pipelines, programs, SBT etc. */
		Renderer();

		void render();

		void resize(const glm::ivec2& newSize);

		// Download rendered color buffer from device
		void downloadPixels(uint32_t h_pixels[]);

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
	};
}