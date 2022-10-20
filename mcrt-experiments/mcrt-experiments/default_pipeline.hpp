#pragma once
#include "mcrt_pipeline.hpp"

namespace mcrt {

	class DefaultPipeline : public McrtPipeline
	{
	public:
		DefaultPipeline(OptixDeviceContext& context, GeometryBufferHandle& geometryBuffers, Scene& scene);

	private:
		void buildModule(OptixDeviceContext& context) override;
		void buildDevicePrograms(OptixDeviceContext& context) override;
		void buildSBT(GeometryBufferHandle& geometryBuffers, Scene& scene) override;
		void buildPipeline(OptixDeviceContext& context) override;
		OptixTraversableHandle buildAccelerationStructure(OptixDeviceContext& context, GeometryBufferHandle& geometryBuffers, Scene& scene) override;
	};

}

