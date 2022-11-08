#pragma once

#include <optix_device.h>
#include "random.hpp"
#include "vec_math.hpp"

#include "LaunchParams.hpp"
#include "glm/glm.hpp"

#define PI 3.14159265358979323846f
#define EPSILON 0.0000000000002f

using namespace mcrt;

namespace mcrt {

    extern "C" __constant__ LaunchParamsRadianceCellGather optixLaunchParams;

    static __forceinline__ __device__
        void* unpackPointer(uint32_t i0, uint32_t i1)
    {
        const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
        void* ptr = reinterpret_cast<void*>(uptr);
        return ptr;
    }

    static __forceinline__ __device__
        void  packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
    {
        const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
        i0 = uptr >> 32;
        i1 = uptr & 0x00000000ffffffff;
    }

    template<typename T>
    static __forceinline__ __device__ T* getPRD()
    {
        const uint32_t u0 = optixGetPayload_0();
        const uint32_t u1 = optixGetPayload_1();
        return reinterpret_cast<T*>(unpackPointer(u0, u1));
    }


    extern "C" __global__ void __closesthit__radiance__cell__gathering()
    {
    }

    extern "C" __global__ void __anyhit__radiance__cell__gathering() {
        // Do nothing
        printf("Hit!HEHEE");
    }

    extern "C" __global__ void __miss__radiance__cell__gathering()
    {

    }

    extern "C" __global__ void __raygen__renderFrame__cell__gathering()
    {
        // Get thread indices
        const int uIndex = optixGetLaunchIndex().x;
        const int vIndex = optixGetLaunchIndex().y;

        // TODO: SKIP PIXELS THAT ARE BLACK!
        uint32_t lightSrcColor = optixLaunchParams.lightSourceTexture.colorBuffer[vIndex * optixLaunchParams.lightSourceTexture.size + uIndex];
        //printf("%d", lightSrcColor);

        glm::vec3 UVWorldPos = optixLaunchParams.uvWorldPositions.UVDataBuffer[vIndex * optixLaunchParams.lightSourceTexture.size + uIndex].worldPosition;
        const glm::vec3 UVNormal = optixLaunchParams.uvWorldPositions.UVDataBuffer[vIndex * optixLaunchParams.lightSourceTexture.size + uIndex].worldNormal;
        // We apply a small offset of 0.00001f in the direction of the normal to the UV world pos, to 'mitigate' floating point rounding errors causing false occlusions/illuminations
        UVWorldPos = glm::vec3{ UVWorldPos.x + UVNormal.x * 0.00001f, UVWorldPos.y + UVNormal.y * 0.00001f, UVWorldPos.z + UVNormal.z * 0.00001f };
        
        // Iterate over all non-empty cells
        for (int i = 0; i < optixLaunchParams.nonEmptyCells.size; i++)
        {
            glm::vec3 cellCenter = optixLaunchParams.nonEmptyCells.centers[i];
            glm::vec3 lightToCellDir = { cellCenter.x - UVWorldPos.x, cellCenter.y - UVWorldPos.y, cellCenter.z - UVWorldPos.z };

            float3 rayOrigin3f = float3{ UVWorldPos.x, UVWorldPos.y, UVWorldPos.z };
            float3 rayDir3f = float3{ lightToCellDir.x, lightToCellDir.y, lightToCellDir.z };

            optixTrace(optixLaunchParams.traversable,
                rayOrigin3f,
                rayDir3f,
                0.f,    // tmin
                1e20f,  // tmax
                0.0f,   // rayTime
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,
                0,  // SBT offset
                1,  // SBT stride
                0  // missSBTIndex 
                );
        }
    }
}