#pragma once

#include <optix_device.h>
#include "random.hpp"
#include "vec_math.hpp"

#include "LaunchParams.hpp"
#include "glm/glm.hpp"

#include "cube_mapping.cuh"

using namespace mcrt;

namespace mcrt {

    extern "C" __constant__ LaunchParamsRadianceCellGatherCubeMap optixLaunchParams;


    static __forceinline__ __device__ RadianceCellGatherPRDAlt loadRadianceCellGatherPRD()
    {
        RadianceCellGatherPRDAlt prd = {};

        prd.resultColor.x = __uint_as_float(optixGetPayload_0());
        prd.resultColor.y = __uint_as_float(optixGetPayload_1());
        prd.resultColor.z = __uint_as_float(optixGetPayload_2());

        return prd;
    }

    static __forceinline__ __device__ void storeRadianceCellGatherPRD(RadianceCellGatherPRDAlt prd)
    {
        optixSetPayload_0(__float_as_uint(prd.resultColor.x));
        optixSetPayload_1(__float_as_uint(prd.resultColor.y));
        optixSetPayload_2(__float_as_uint(prd.resultColor.z));
    }


    extern "C" __global__ void __closesthit__radiance__cell__gathering__scene()
    {
        const MeshSBTDataRadianceCellGather& sbtData
            = *(const MeshSBTDataRadianceCellGather*)optixGetSbtDataPointer();

        const int   primID = optixGetPrimitiveIndex();
        const glm::ivec3 index = sbtData.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        // Barycentric tex coords
        const glm::vec2 tc
            = (1.f - u - v) * sbtData.texcoord[index.x]
            + u * sbtData.texcoord[index.y]
            + v * sbtData.texcoord[index.z];

        float4 incomingRadiance = tex2D<float4>(optixLaunchParams.lightSourceTextures[sbtData.objectNr], tc.x, tc.y);

        RadianceCellGatherPRDAlt prd = loadRadianceCellGatherPRD();
        prd.resultColor = glm::vec3{ incomingRadiance.x, incomingRadiance.y, incomingRadiance.z };

        storeRadianceCellGatherPRD(prd);
    }

    extern "C" __global__ void __anyhit__radiance__cell__gathering__scene() {
        // Do nothing
        printf("Hit scene!");
    }

    extern "C" __global__ void __miss__radiance__cell__gathering()
    {
        RadianceCellGatherPRDAlt prd = loadRadianceCellGatherPRD();
        prd.resultColor = glm::vec3{ 0.0f, 0.0f, 0.0f };
        storeRadianceCellGatherPRD(prd);
    }


    extern "C" __global__ void __raygen__renderFrame__cell__gathering()
    {
        // Get thread indices
        const glm::vec3 probePos = optixLaunchParams.probePosition;

        // Light source texture tiling
        const int uPixel = optixGetLaunchIndex().x;
        const int vPixel = optixGetLaunchIndex().y;
        const int faceIndex = optixGetLaunchIndex().z;

        const float u = float(uPixel) / float(optixLaunchParams.cubeMapResolution);
        const float v = float(vPixel) / float(optixLaunchParams.cubeMapResolution);

        // Offset into cubemap face array
        int probeOffset = optixLaunchParams.probeOffset;

        // Ray direction (convert cubemap space to 3D direction vector)
        glm::vec3 rayDir;
        convert_uv_to_cube_xyz(faceIndex, u, v, &rayDir.x, &rayDir.y, &rayDir.z);

        // Convert to float3 format
        float3 rayOrigin3f = float3{ probePos.x, probePos.y, probePos.z };
        float3 rayDir3f = float3{ rayDir.x, rayDir.y, rayDir.z };
        float3 normalizedRayDir = normalize(rayDir3f);

        RadianceCellGatherPRDAlt prd{};

        unsigned int u0, u1, u2;

        // Trace ray against scene geometry to see if ray is occluded
        optixTrace(optixLaunchParams.sceneTraversable,
            rayOrigin3f,
            normalizedRayDir,
            0.f,    // tmin
            1e20f,  // tmax
            0.0f,   // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,      // We only need closest-hit for scene geometry
            0,  // SBT offset
            1,  // SBT stride
            0,  // missSBTIndex
            u0, u1, u2
        );

        // Write rgb
        float4 resultValue = float4{ __uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2), 0.0f };
        surf2Dwrite(resultValue, optixLaunchParams.cubeMaps[probeOffset + faceIndex], uPixel * 16, vPixel);

    }
}