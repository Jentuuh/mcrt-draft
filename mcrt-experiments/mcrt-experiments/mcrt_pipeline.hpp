#pragma once
#include "CUDABuffer.hpp"
#include "LaunchParams.hpp"
#include "scene.hpp"

namespace mcrt {
	struct GeometryBufferHandle {
		std::vector<CUDABuffer>& vertices;
		std::vector<CUDABuffer>& indices;
		std::vector<CUDABuffer>& normals;
		std::vector<CUDABuffer>& texCoords;
		std::vector<cudaTextureObject_t> &textureObjects;
	};

	class McrtPipeline
	{
	public:
		McrtPipeline(OptixDeviceContext& context, GeometryBufferHandle& geometryBuffers, Scene& scene);

		virtual void uploadLaunchParams() = 0;

		OptixPipeline				pipeline;

		// SBT
		OptixShaderBindingTable sbt = {};


	private:
		virtual void buildModule(OptixDeviceContext& context) = 0;
		virtual void buildDevicePrograms(OptixDeviceContext& context) = 0;
		virtual void buildSBT(GeometryBufferHandle& geometryBuffers, Scene& scene) = 0;
		virtual void buildPipeline(OptixDeviceContext& context) = 0;
		virtual OptixTraversableHandle buildAccelerationStructure(OptixDeviceContext& context, GeometryBufferHandle& geometryBuffers, Scene& scene) = 0;

	protected:
		void init(OptixDeviceContext& context, GeometryBufferHandle& geometryBuffers, Scene& scene);

		// Pipeline + properties
		OptixPipelineCompileOptions	pipelineCompileOptions = {};
		OptixPipelineLinkOptions	pipelineLinkOptions = {};

		// Module containing device programs
		OptixModule					module;
		OptixModuleCompileOptions	moduleCompileOptions = {};

		// Vectors of all program(group)s
		std::vector<OptixProgramGroup> raygenPGs;
		CUDABuffer raygenRecordsBuffer;
		std::vector<OptixProgramGroup> missPGs;
		CUDABuffer missRecordsBuffer;
		std::vector<OptixProgramGroup> hitgroupPGs;
		CUDABuffer hitgroupRecordsBuffer;

		// AS
		CUDABuffer accelerationStructBuffer;
	};
}


