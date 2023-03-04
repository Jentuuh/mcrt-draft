#pragma once

namespace mcrt {
	class IntersectionTester
	{
	public:
		IntersectionTester();
		static bool triangleBoxOverlap(float boxcenter[3], float boxhalfsize[3], float** triverts);
		static bool planeBoxOverlap(float normal[3], float d, float maxbox[3]);
	};
}

