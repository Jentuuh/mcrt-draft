#pragma once
#include <string>
#include <vector>

namespace mcrt {

	class HDRImage
	{
	public:
		HDRImage(int width, int height, float* data);

		void saveImage(std::string fileName);

		int width;
		int height;

	private:
		float* pixels;
		int numChannels;
	};

}

