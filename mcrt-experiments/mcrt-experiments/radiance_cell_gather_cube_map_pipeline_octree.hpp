#pragma once
#include "mcrt_pipeline.hpp"

namespace mcrt {
	class RadianceCellGatherCubeMapPipelineOctree : public McrtPipeline
	{
	public:
		RadianceCellGatherCubeMapPipelineOctree(OptixDeviceContext& context, GeometryBufferHandle& proxyGeometry, Scene& scene);
		void uploadLaunchParams() override;

		LaunchParamsRadianceCellGatherCubeMapOctree launchParams;
		CUDABuffer   launchParamsBuffer;

	private:
		void initRadianceCellGather(OptixDeviceContext& context, GeometryBufferHandle& proxyGeometry, Scene& scene);
		void buildModule(OptixDeviceContext& context) override;
		void buildDevicePrograms(OptixDeviceContext& context) override;
		void buildSBT(GeometryBufferHandle& geometryBuffers, Scene& scene) override;
		void buildSBTRadianceCellGather(GeometryBufferHandle& proxyGeometry, Scene& scene);
		void buildPipeline(OptixDeviceContext& context) override;
		OptixTraversableHandle buildAccelerationStructure(OptixDeviceContext& context, GeometryBufferHandle& geometryBuffers, Scene& scene) override;
	};
}
