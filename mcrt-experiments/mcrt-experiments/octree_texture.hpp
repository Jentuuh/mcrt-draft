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
	private:
		void initOctreeTexture(int d, Scene& sceneObject);

		CUDABuffer octreeGPUMemory;
		//std::vector<cudaTextureObject_t> textureObjects;
	};
}
