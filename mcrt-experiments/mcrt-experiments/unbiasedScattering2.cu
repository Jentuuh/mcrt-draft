#pragma once

#include <optix_device.h>
#include "random.hpp"
#include "vec_math.hpp"

#include "LaunchParams.hpp"
#include "glm/glm.hpp"

#include "cube_mapping.cuh"
#include "utils.cuh"

#define PI 3.14159265358979323846f
#define EPSILON 0.0000000000002f
#define NUM_DIRECTION_SAMPLES 5000
#define PI_OVER_4 0.785398163397425f
#define PI_OVER_2 1.5707963267945f

using namespace mcrt;


namespace mcrt {
    extern "C" __constant__ LaunchParamsRadianceCellScatterUnbiased optixLaunchParams;

    static __forceinline__ __device__ RadianceCellScatterPRD loadRadianceCellScatterPRD()
    {
        RadianceCellScatterPRD prd = {};

        prd.distanceToClosestIntersectionSquared = __uint_as_float(optixGetPayload_0());
        prd.rayOrigin.x = __uint_as_float(optixGetPayload_1());
        prd.rayOrigin.y = __uint_as_float(optixGetPayload_2());
        prd.rayOrigin.z = __uint_as_float(optixGetPayload_3());

        return prd;
    }

    static __forceinline__ __device__ void storeRadianceCellScatterPRD(RadianceCellScatterPRD prd)
    {
        optixSetPayload_0(__float_as_uint(prd.distanceToClosestIntersectionSquared));
        optixSetPayload_1(__float_as_uint(prd.rayOrigin.x));
        optixSetPayload_2(__float_as_uint(prd.rayOrigin.y));
        optixSetPayload_3(__float_as_uint(prd.rayOrigin.z));
    }


    extern "C" __global__ void __closesthit__radiance__cell__scattering__scene__unbiased()
    {
        const MeshSBTDataRadianceCellScatter& sbtData
            = *(const MeshSBTDataRadianceCellScatter*)optixGetSbtDataPointer();

        const int primID = optixGetPrimitiveIndex();
        const glm::ivec3 index = sbtData.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        const glm::vec3 intersectionWorldPos =
            (1.f - u - v) * sbtData.vertex[index.x]
            + u * sbtData.vertex[index.y]
            + v * sbtData.vertex[index.z];

        RadianceCellScatterPRD prd = loadRadianceCellScatterPRD();
        float distanceToIntersectionSquared = (((intersectionWorldPos.x - prd.rayOrigin.x) * (intersectionWorldPos.x - prd.rayOrigin.x)) + ((intersectionWorldPos.y - prd.rayOrigin.y) * (intersectionWorldPos.y - prd.rayOrigin.y)) + ((intersectionWorldPos.z - prd.rayOrigin.z) * (intersectionWorldPos.z - prd.rayOrigin.z)));

        prd.distanceToClosestIntersectionSquared = distanceToIntersectionSquared;
        storeRadianceCellScatterPRD(prd);
    }

    extern "C" __global__ void __anyhit__radiance__cell__scattering__scene__unbiased() {
        // Do nothing
    }

    extern "C" __global__ void __miss__radiance__cell__scattering__unbiased()
    {
        // Do nothing
    }

    extern "C" __global__ void __raygen__renderFrame__cell__scattering__unbiased()
    {
        const int uvIndex = optixGetLaunchIndex().x;

        const int nonEmptyCellIndex = optixLaunchParams.nonEmptyCellIndex;

        // Take different seed for each radiance cell face
        unsigned int seed = tea<4>(uvIndex, nonEmptyCellIndex);

        // Get UV world position for this shader pass
        const int uvInsideOffset = optixLaunchParams.uvsInsideOffsets[nonEmptyCellIndex];
        glm::vec2 uv = optixLaunchParams.uvsInside[uvInsideOffset + uvIndex];
        const int u = int(uv.x * optixLaunchParams.uvWorldPositions.size);
        const int v = int(uv.y * optixLaunchParams.uvWorldPositions.size);

        /*if (nonEmptyCellIndex == 0)
        {
            printf("(%f, %f)\n", uv.x, uv.y);
        }*/

        glm::vec3 UVWorldPos = optixLaunchParams.uvWorldPositions.UVDataBuffer[v * optixLaunchParams.uvWorldPositions.size + u].worldPosition;
        const glm::vec3 UVNormal = optixLaunchParams.uvWorldPositions.UVDataBuffer[v * optixLaunchParams.uvWorldPositions.size + u].worldNormal;
        const glm::vec3 diffuseColor = optixLaunchParams.uvWorldPositions.UVDataBuffer[v * optixLaunchParams.uvWorldPositions.size + u].diffuseColor;

        // We apply a small offset of 0.00001f in the direction of the normal to the UV world pos, to 'mitigate' floating point rounding errors causing false occlusions/illuminations
        UVWorldPos = glm::vec3{ UVWorldPos.x + UVNormal.x * 0.0000001f, UVWorldPos.y + UVNormal.y * 0.0000001f, UVWorldPos.z + UVNormal.z * 0.0000001f };

        float3 rayOrigin3f = float3{ UVWorldPos.x, UVWorldPos.y, UVWorldPos.z };
        float3 uvNormal3f = float3{ UVNormal.x, UVNormal.y, UVNormal.z };

        //// ==============================================================================
        //// Calculate rotation matrix to align generated directions with normal hemisphere
        //// ==============================================================================
        //float3 up = float3{ 0.0f, 1.0f, 0.0f };
        //glm::mat3x3 rotationMatrix;
        //if (uvNormal3f.x == 0.0f && uvNormal3f.y == -1.0f && uvNormal3f.z == 0.0f)
        //{
        //    float3 rotAxis = cross(up, uvNormal3f);
        //    float sine = length(rotAxis);
        //    float cosine = dot(up, uvNormal3f);
        //    glm::mat3x3 v_x = glm::mat3x3{ {0.0f, rotAxis.z, -rotAxis.y}, {-rotAxis.z, 0.0f, rotAxis.x}, {rotAxis.y, -rotAxis.x, 0.0f} };
        //    glm::mat3x3 v_xSquared = v_x * v_x;
        //    float endFactor = 1.0f / (1 + cosine);
        //    rotationMatrix = glm::mat3x3(1.0f) + v_x + (v_xSquared * endFactor);
        //}
        //else {
        //    // 180 degrees rotation around the X-axis (flips along Y-axis)
        //    rotationMatrix = glm::mat3x3{ {1.0f, 0.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, -1.0f} };
        //}



        // ======================================
        // Radiance + num of samples accumulators
        // ======================================
        glm::vec3 totalRadiance = glm::vec3{ 0.0f, 0.0f, 0.0f };
        int numSamples = 0;

        for (int i = 0; i < NUM_DIRECTION_SAMPLES; i++)
        {

            // =================================================================================================================================================================================
            // Random direction generation (uniform direction generation with spherical coords)  : https://math.stackexchange.com/questions/44689/how-to-find-a-random-axis-or-unit-vector-in-3d
            // =================================================================================================================================================================================

            bool dirFound = false;

            float2 uniformRandoms = float2{ rnd(seed), rnd(seed) };
            glm::ivec2 uvLightSrc = glm::ivec2{ int(uniformRandoms.x * optixLaunchParams.prevBounceTexture.size), int(uniformRandoms.y * optixLaunchParams.prevBounceTexture.size) };
            glm::vec3 lightSrcWorldPos = optixLaunchParams.uvWorldPositions.UVDataBuffer[uvLightSrc.y * optixLaunchParams.uvWorldPositions.size + uvLightSrc.x].worldPosition;

            uint32_t lightSrcColor = optixLaunchParams.prevBounceTexture.colorBuffer[uvLightSrc.y * optixLaunchParams.prevBounceTexture.size + uvLightSrc.x];

            // Extract rgb values from light source texture pixel
            uint32_t rLightSrc = 0x000000ff & (lightSrcColor);
            uint32_t gLightSrc = (0x0000ff00 & (lightSrcColor)) >> 8;
            uint32_t bLightSrc = (0x00ff0000 & (lightSrcColor)) >> 16;

            glm::vec3 rayDir = lightSrcWorldPos - UVWorldPos;
            float3 rayDir3f = float3{ rayDir.x, rayDir.y, rayDir.z };

            if (dot(rayDir3f, uvNormal3f) < 0)
            {
                continue;
            }
            

            RadianceCellScatterPRD prd;
            unsigned int u0, u1, u2, u3;

            u1 = __float_as_uint(prd.rayOrigin.x);
            u2 = __float_as_uint(prd.rayOrigin.y);
            u3 = __float_as_uint(prd.rayOrigin.z);

            // Trace ray against scene geometry to see if ray is occluded
            optixTrace(optixLaunchParams.sceneTraversable,
                rayOrigin3f,
                rayDir3f,
                0.f,    // tmin
                1e20f,  // tmax
                0.0f,   // rayTime
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,      // We only need closest-hit for scene geometry
                0,  // SBT offset
                1,  // SBT stride
                0,  // missSBTIndex
                u0, u1, u2, u3
            );
            
            prd.distanceToClosestIntersectionSquared = __uint_as_float(u0);
            float distanceToUVSquared = (((UVWorldPos.x - lightSrcWorldPos.x) * (UVWorldPos.x - lightSrcWorldPos.x)) + ((UVWorldPos.y - lightSrcWorldPos.y) * (UVWorldPos.y - lightSrcWorldPos.y)) + ((UVWorldPos.z - lightSrcWorldPos.z) * (UVWorldPos.z - lightSrcWorldPos.z)));

            if (distanceToUVSquared < prd.distanceToClosestIntersectionSquared)
            {
                // Cosine weighted contribution
                float cosContribution = dot(rayDir3f, uvNormal3f);
                totalRadiance += glm::vec3{ cosContribution * rLightSrc, cosContribution * gLightSrc, cosContribution * bLightSrc };
                ++numSamples;
            }
        }

        // TODO: add Monte Carlo weight (although how do we get the exact pdf that we are sampling from...?)
        const int r_result = int((totalRadiance.x / (float(numSamples))));
        const int g_result = int((totalRadiance.y / (float(numSamples))));
        const int b_result = int((totalRadiance.z / (float(numSamples))));

        //if (totalRadiance.x > 0.0f || totalRadiance.y > 0.0f || totalRadiance.z > 0.0f)
        //{
        //    printf("Total contribution: %d, %d, %d\n", r_result, g_result, b_result);
        //}


        // convert to 32-bit rgba value (we explicitly set alpha to 0xff
        // to make stb_image_write happy ...
        const uint32_t rgba = 0xff000000
            | (r_result << 0) | (g_result << 8) | (b_result << 16);

        optixLaunchParams.currentBounceTexture.colorBuffer[v * optixLaunchParams.currentBounceTexture.size + u] = rgba;
    }
}