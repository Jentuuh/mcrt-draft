#pragma once
#include <optix_device.h>
#include "LaunchParams.hpp"


using namespace mcrt;

namespace mcrt {


	
	static __forceinline__ __device__ void convert_xyz_to_cube_uv(float x, float y, float z, int* index, float* u, float* v)
	{
        float maxExtent = max(max(abs(x), abs(y)), abs(z));
        int faceIndex;
        float uc;
        float vc;

        // X axis is max extent
        if (maxExtent == abs(x))
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
        else if (maxExtent == abs(y))
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
        else if (maxExtent == abs(z))
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
        *u = 0.5 * ((uc / maxExtent) + 1.0);
        *v = 0.5 * ((vc / maxExtent) + 1.0);
        *index = faceIndex;
	}


    static __forceinline__ __device__ void convert_uv_to_cube_xyz(int index, float u, float v, float* x, float* y, float* z)
    {
        // convert range 0 to 1 to -1 to 1
        float uc = 2.0f * u - 1.0f;
        float vc = 2.0f * v - 1.0f;
        switch (index)
        {
        case 0: *x = 1.0f; *y = vc; *z = -uc; break;	// POSITIVE X
        case 1: *x = -1.0f; *y = vc; *z = uc; break;	// NEGATIVE X
        case 2: *x = uc; *y = 1.0f; *z = -vc; break;	// POSITIVE Y
        case 3: *x = uc; *y = -1.0f; *z = vc; break;	// NEGATIVE Y
        case 4: *x = uc; *y = vc; *z = 1.0f; break;	// POSITIVE Z
        case 5: *x = -uc; *y = vc; *z = -1.0f; break;	// NEGATIVE Z
        }
    }
}