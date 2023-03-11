#pragma once

#include "LaunchParams.hpp"
#include "CUDABuffer.hpp"
#include "octree_builder.hpp"
#include "scene.hpp"
#include <vector>


namespace mcrt {
	class OctreeTexture
	{
	public:
		OctreeTexture(int d, Scene& sceneObject);

		//std::vector<cudaTextureObject_t>& getTextureObjects() { return textureObjects; }; 
		CUDABuffer& getOctreeGPUMemory() { return octreeGPUMemory; }; 
		CUDABuffer& getOctreeGPUMemoryBounce2() { return octreeGPUMemoryBounce2; };

		//std::vector<glm::vec3>& getLeafPositions() { return leafPositions; };
		float getLeafFaceArea() { return leafFaceArea; };
		int getKernelGranularity() { return kernelGranularity; };

	private:
		void initOctreeTexture(int d, Scene& sceneObject);
		CUDABuffer octreeGPUMemory;
		CUDABuffer octreeGPUMemoryBounce2;

		int kernelGranularity;
		float leafFaceArea;
	};
}
