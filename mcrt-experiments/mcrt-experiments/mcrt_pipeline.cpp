#include "mcrt_pipeline.hpp"


namespace mcrt {

	McrtPipeline::McrtPipeline(OptixDeviceContext& context, GeometryBufferHandle& geometryBuffers, Scene& scene){	}

	void McrtPipeline::init(OptixDeviceContext& context, GeometryBufferHandle& geometryBuffers, Scene& scene)
	{
		buildModule(context);
		buildDevicePrograms(context);
		buildPipeline(context);
		buildSBT(geometryBuffers, scene);
	}


	void McrtPipeline::uploadLaunchParams()
	{
	}
}