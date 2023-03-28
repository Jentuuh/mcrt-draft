#include "hdr_image.hpp"
#include <stb/stb_image_write.h>

namespace mcrt {
	HDRImage::HDRImage(int width, int height, float* data):width{width}, height{height}, pixels{data}
	{}

	void HDRImage::saveImage(std::string fileName)
	{
		stbi_write_hdr(fileName.c_str(), width, height, 4, pixels);
	}
}