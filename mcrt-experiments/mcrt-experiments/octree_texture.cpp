#include "octree_texture.hpp"
#include <iostream>

namespace mcrt {

	OctreeTexture::OctreeTexture(int d, Scene& sceneObject)
	{
		initOctreeTexture(d, sceneObject);
	}


	void OctreeTexture::initOctreeTexture(int d, Scene& sceneObject)
	{
		int resolution = 1024;
		leafFaceArea = (1.0f / powf(2, d)) * (1.0f / powf(2, d));

		OctreeBuilder builder{ d, resolution, sceneObject };
		//OctreeBuilder builder{ "../data/octreeSaveFile.txt" };

		//builder.saveOctreeToFile("../data/octreeSaveFile.txt");
		//builder.writeGPUOctreeToTxtFile("../debug_output/octreeTexture.txt");

		// Upload to GPU
		octreeGPUMemory.alloc_and_upload(builder.getOctree());
		octreeGPUMemoryBounce2.alloc_and_upload(builder.getOctree());
	}

}