#include "octree_texture.hpp"
#include <iostream>

namespace mcrt {

	OctreeTexture::OctreeTexture(int d)
	{
		initOctreeTexture(d);
	}

	void OctreeTexture::initOctreeTexture(int d)
	{
		textureObjects.resize(1);

		std::cout << "Initializing octree textures..." << std::endl;
		int resolution = pow(2, d);
		std::cout << "Octree resolution: " << resolution << std::endl;
		cudaResourceDesc res_desc = {};

		cudaChannelFormatDesc channel_desc;
		int32_t width = resolution;
		int32_t height = resolution;
		int32_t depth = resolution;
		int32_t numComponents = 4; // RGBA (alpha channel used to determine if node contains leaf data, child ptr or empty)
		int32_t pitch = width * numComponents * sizeof(float);

		std::vector<float> initializationVector(width * height * depth * 4, 0.0f);
		// TODO: build octree data structure into initialization vector

		channel_desc = cudaCreateChannelDesc<float>(); // 4 8-bit channels

		// Array creation + copy data to array
		cudaArray* pixelArray;
		CUDA_CHECK(Malloc3DArray(&pixelArray, &channel_desc, make_cudaExtent(pitch, height, depth), 0));
		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = make_cudaPitchedPtr(initializationVector.data(), pitch, height, depth);
		copyParams.dstArray = pixelArray;
		copyParams.extent = make_cudaExtent(width, height, depth);
		copyParams.kind = cudaMemcpyHostToDevice;
		CUDA_CHECK(Memcpy3D(&copyParams));

		cudaResourceDesc    textureResource;
		memset(&textureResource, 0, sizeof(cudaResourceDesc));
		textureResource.resType = cudaResourceTypeArray;
		textureResource.res.array.array = pixelArray;
		cudaTextureDesc textureDescription;
		memset(&textureDescription, 0, sizeof(cudaTextureDesc));
		textureDescription.normalizedCoords = false;
		textureDescription.filterMode = cudaFilterModePoint;
		textureDescription.addressMode[0] = cudaAddressModeClamp;
		textureDescription.addressMode[1] = cudaAddressModeClamp;
		textureDescription.addressMode[2] = cudaAddressModeClamp;
		textureDescription.readMode = cudaReadModeElementType;

		CUDA_CHECK(CreateTextureObject(&textureObjects[0], &textureResource, &textureDescription, NULL));
	}

}