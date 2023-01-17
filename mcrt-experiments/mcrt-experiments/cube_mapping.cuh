#include <optix_device.h>
#include "LaunchParams.hpp"


using namespace mcrt;

namespace mcrt {


	
	static __forceinline__ __device__ void convert_xyz_to_cube_uv(float x, float y, float z, int* index, float* u, float* v)
	{
        float maxExtent = max(max(x, y), z);
        int faceIndex;
        float uc;
        float vc;

        // X axis is max extent
        if (maxExtent == x)
        {
            // Positive X
            if (x > 0)
            {
                faceIndex = 0;
                uc = -z;
                vc = y;
            }
            // Negative X
            else {
                faceIndex = 1;
                uc = z;
                vc = y;
            }
        }
        // Y axis is max extent
        else if (maxExtent == y)
        {
            // Positive Y
            if (y > 0)
            {
                uc = x;
                vc = -z;
                faceIndex = 2;
            }
            // Negative Y
            else {
                uc = x;
                vc = z;
                faceIndex = 3;
            }
        }
        // Z axis is max extent
        else if (maxExtent == z)
        {
            // Positive Z
            if (z > 0)
            {
                uc = x;
                vc = y;
                faceIndex = 4;
            }
            // Negative Z
            else {
                uc = -x;
                vc = y;
                faceIndex = 5;
            }
        }

        // Shift from [-1; 1] to [0; 1]
        *u = 0.5 * (uc / maxExtent + 1.0);
        *v = 0.5 * (vc / maxExtent + 1.0);
        *index = faceIndex;
	}
}