#include "mcrt_pipeline.hpp"


namespace mcrt {

	McrtPipeline::McrtPipeline(OptixDeviceContext& context, GeometryBufferHandle& geometryBuffers, Scene& scene)
	{
		//init(context, geometryBuffers, scene);
	}

	void McrtPipeline::init(OptixDeviceContext& context, GeometryBufferHandle& geometryBuffers, Scene& scene)
	{
		buildModule(context);
		buildDevicePrograms(context);
		launchParams.traversable = buildAccelerationStructure(context, geometryBuffers, scene);
		buildPipeline(context);
		buildSBT(geometryBuffers, scene);

		launchParamsBuffer.alloc(sizeof(launchParams));
	}


	void McrtPipeline::uploadLaunchParams()
	{
		launchParamsBuffer.upload(&launchParams, 1);
	}
}