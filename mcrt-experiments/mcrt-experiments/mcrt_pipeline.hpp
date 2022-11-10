#pragma once
#include "CUDABuffer.hpp"
#include "LaunchParams.hpp"
#include "scene.hpp"
#include "gas.hpp"

namespace mcrt {


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
		void buildGASes(OptixDeviceContext& context, std::vector<GeometryBufferHandle&> geometries, std::vector<int> numsBuildInputs);

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

		// GASes
		std::vector<GAS> GASes;

		// AS
		CUDABuffer accelerationStructBuffer;
	};
}


