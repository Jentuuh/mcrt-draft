#pragma once
#include "CUDABuffer.hpp"
#include "LaunchParams.hpp"
#include "gas.hpp"
#include <iostream>

namespace mcrt {
	class IAS
	{
	public:
		IAS(OptixDeviceContext& context, std::vector<glm::mat4> transforms, std::vector<GAS> gases, int numRayTypes);

		void build(OptixDeviceContext& context, const std::vector<OptixInstance>& instances);

		OptixTraversableHandle traversableHandle() { return iasTraversableHandle; };
	private:
		CUDABuffer accelerationStructBuffer;
		CUDABuffer d_instances;
		OptixTraversableHandle iasTraversableHandle;
	};
}


