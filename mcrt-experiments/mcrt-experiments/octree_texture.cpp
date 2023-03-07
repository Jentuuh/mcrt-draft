#include "octree_texture.hpp"
#include <iostream>

namespace mcrt {

	OctreeTexture::OctreeTexture(int d, Scene& sceneObject)
	{
		initOctreeTexture(d, sceneObject);
	}

	//void OctreeTexture::initOctreeTexture(int d, Scene& sceneObject)
	//{
	//	textureObjects.resize(1);

	//	int resolution = 1024;


	//	// TODO: build octree data structure into initialization vector
	//	OctreeBuilder builder{ d, resolution, sceneObject};

	//	std::cout << "Initializing octree textures..." << std::endl;
	//	cudaResourceDesc res_desc = {};

	//	int32_t width = resolution;
	//	int32_t height = builder.getTextureDimensions().z == 0 ? builder.getTextureDimensions().y + 1 : resolution;
	//	int32_t depth = builder.getTextureDimensions().z + 1;
	//	int32_t pitch = width * sizeof(float);
	//	
	//	// =============================================================================================================
	//	// Allocate linear source memory on GPU as well, to ensure pitch compatibility with hardware
	//	// See https://stackoverflow.com/questions/10611451/how-to-use-make-cudaextent-to-define-a-cudaextent-correctly 
	//	// =============================================================================================================
	//	cudaExtent volumeSizeBytes = make_cudaExtent(pitch, height, depth);
	//	cudaPitchedPtr d_volumeMem;
	//	CUDA_CHECK(Malloc3D(&d_volumeMem, volumeSizeBytes));

	//	size_t size = d_volumeMem.pitch * height * depth;
	//	float* h_volumeMem = builder.getOctree().data();
	//	CUDA_CHECK(Memcpy(d_volumeMem.ptr, h_volumeMem, size, cudaMemcpyHostToDevice));

	//	cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();

	//	// ======================
	//	// Device to device copy
	//	// ======================
	//	cudaArray* pixelArray;
	//	CUDA_CHECK(Malloc3DArray(&pixelArray, &channel_desc, make_cudaExtent(width, height, depth), 0));
	//	cudaMemcpy3DParms copyParams = { 0 };
	//	copyParams.srcPtr = d_volumeMem;
	//	copyParams.dstArray = pixelArray;
	//	copyParams.extent = make_cudaExtent(width, height, depth);
	//	copyParams.kind = cudaMemcpyDeviceToDevice;
	//	CUDA_CHECK(Memcpy3D(&copyParams));

	//	// =============================
	//	// Creation of texture resource
	//	// =============================
	//	cudaResourceDesc    textureResource;
	//	memset(&textureResource, 0, sizeof(cudaResourceDesc));
	//	textureResource.resType = cudaResourceTypeArray;
	//	textureResource.res.array.array = pixelArray;
	//	cudaTextureDesc textureDescription;
	//	memset(&textureDescription, 0, sizeof(cudaTextureDesc));
	//	textureDescription.normalizedCoords = false;
	//	textureDescription.filterMode = cudaFilterModePoint;
	//	textureDescription.addressMode[0] = cudaAddressModeClamp;
	//	textureDescription.addressMode[1] = cudaAddressModeClamp;
	//	textureDescription.addressMode[2] = cudaAddressModeClamp;
	//	textureDescription.readMode = cudaReadModeElementType;

	//	CUDA_CHECK(CreateTextureObject(&textureObjects[0], &textureResource, &textureDescription, NULL));
	//	std::cout << "Octree texture created." << std::endl;

	//}

	void OctreeTexture::initOctreeTexture(int d, Scene& sceneObject)
	{
		int resolution = 1024;

		// TODO: build octree data structure into initialization vector
		OctreeBuilder builder{ d, resolution, sceneObject };

		leafPositions = builder.getLeafPositions();
		kernelGranularity = sqrt(leafPositions.size());
		std::cout << "Kernel granularity: " << kernelGranularity << std::endl;

		//builder.writeGPUOctreeToTxtFile("../debug_output/octreeTexture.txt");

		// Upload to GPU
		octreeGPUMemory.alloc_and_upload(builder.getOctree());
	}

}