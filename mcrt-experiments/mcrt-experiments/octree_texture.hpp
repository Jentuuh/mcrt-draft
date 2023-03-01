#pragma once

#include "LaunchParams.hpp"
#include <vector>


namespace mcrt {
	class OctreeTexture
	{
	public:
		OctreeTexture(int d);

		std::vector<cudaTextureObject_t>& getTextureObjects() { return textureObjects; }; 
	private:
		void initOctreeTexture(int d);

		std::vector<cudaTextureObject_t> textureObjects;
	};
}
