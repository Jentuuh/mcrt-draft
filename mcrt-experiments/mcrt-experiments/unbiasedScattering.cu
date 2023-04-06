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
#define NUM_DIRECTION_SAMPLES 10
#define PI_OVER_4 0.785398163397425f
#define PI_OVER_2 1.5707963267945f

using namespace mcrt;


namespace mcrt {
    extern "C" __constant__ LaunchParamsRadianceCellScatterUnbiased optixLaunchParams;

    static __forceinline__ __device__ RadianceCellScatterUnbiasedPRD loadRadianceCellScatterUnbiasedPRD()
    {
        RadianceCellScatterUnbiasedPRD prd = {};

        prd.resultColor.x = __uint_as_float(optixGetPayload_0());
        prd.resultColor.y = __uint_as_float(optixGetPayload_1());
        prd.resultColor.z = __uint_as_float(optixGetPayload_2());

        return prd;
    }

    static __forceinline__ __device__ void storeRadianceCellScatterUnbiasedPRD(RadianceCellScatterUnbiasedPRD prd)
    {
        optixSetPayload_0(__float_as_uint(prd.resultColor.x));
        optixSetPayload_1(__float_as_uint(prd.resultColor.y));
        optixSetPayload_2(__float_as_uint(prd.resultColor.z));
    }


    extern "C" __global__ void __closesthit__radiance__cell__scattering__scene__unbiased()
    {
        const MeshSBTDataRadianceCellScatter& sbtData
            = *(const MeshSBTDataRadianceCellScatter*)optixGetSbtDataPointer();

        const int primID = optixGetPrimitiveIndex();
        const glm::ivec3 index = sbtData.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        // Barycentric tex coords
        const glm::vec2 tc
            = (1.f - u - v) * sbtData.texcoord[index.x]
            + u * sbtData.texcoord[index.y]
            + v * sbtData.texcoord[index.z];

        // Read color (outgoing radiance) at intersection (NOTE THAT WE ASSUME LAMBERTIAN SURFACE HERE)
        // --> Otherwise BRDF needs to be evaluated for the incoming direction at this point
        float4 incomingRadiance = tex2D<float4>(optixLaunchParams.prevBounceTextures[sbtData.objectNr], tc.x, tc.y);

        RadianceCellScatterUnbiasedPRD prd = loadRadianceCellScatterUnbiasedPRD();
        prd.resultColor = glm::vec3{ incomingRadiance.x, incomingRadiance.y, incomingRadiance.z };
        storeRadianceCellScatterUnbiasedPRD(prd);
    }

    extern "C" __global__ void __anyhit__radiance__cell__scattering__scene__unbiased() {
        // Do nothing
    }

    extern "C" __global__ void __miss__radiance__cell__scattering__unbiased()
    {
        RadianceCellScatterUnbiasedPRD prd = loadRadianceCellScatterUnbiasedPRD();
        prd.resultColor = { 0.0f, 0.0f, 0.0f };
        storeRadianceCellScatterUnbiasedPRD(prd);
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
        const int gameObjectNr = optixLaunchParams.uvGameObjectNrs[uvInsideOffset + uvIndex];

        float4 uvWorldPos3f = tex2D<float4>(optixLaunchParams.uvPositions[gameObjectNr], uv.x, uv.y);
        float4 uvWorldNormal3f = tex2D<float4>(optixLaunchParams.uvNormals[gameObjectNr], uv.x, uv.y);
        float4 uvDiffuseColor3f = tex2D<float4>(optixLaunchParams.uvDiffuseColors[gameObjectNr], uv.x, uv.y);

        glm::vec3 UVWorldPos = glm::vec3{ uvWorldPos3f.x, uvWorldPos3f.y, uvWorldPos3f.z };
        const glm::vec3 UVNormal = glm::vec3{ uvWorldNormal3f.x, uvWorldNormal3f.y, uvWorldNormal3f.z };
        glm::vec3 diffuseColor = glm::vec3{ uvDiffuseColor3f.x, uvDiffuseColor3f.y, uvDiffuseColor3f.z };

        if (optixLaunchParams.hasTexture[gameObjectNr])
        {
            // Get diffuse texture coordinates
            float4 diffuseTextureUV = tex2D<float4>(optixLaunchParams.diffuseTextureUVs[gameObjectNr], uv.x, uv.y);

            // Read color from diffuse texture
            float4 diffuseTexColor = tex2D<float4>(optixLaunchParams.diffuseTextures[gameObjectNr], diffuseTextureUV.x, diffuseTextureUV.y);
            diffuseColor = glm::vec3{ diffuseTexColor.x, diffuseTexColor.y, diffuseTexColor.z };
        }

        // Small offset to world position to 'mitigate' floating point errors
        UVWorldPos = glm::vec3{ UVWorldPos.x + UVNormal.x * 0.00001f, UVWorldPos.y + UVNormal.y * 0.00001f, UVWorldPos.z + UVNormal.z * 0.00001f };

        float3 rayOrigin3f = float3{ UVWorldPos.x, UVWorldPos.y, UVWorldPos.z };
        float3 uvNormal3f = float3{ UVNormal.x, UVNormal.y, UVNormal.z };


        // ======================================
        // Radiance + num of samples accumulators
        // ======================================
        glm::vec3 totalRadiance = glm::vec3{ 0.0f, 0.0f, 0.0f };
        int numSamples = 0;

        for (int i = 0; i < NUM_DIRECTION_SAMPLES; i++)
        {
            // =============================================================================================================================================================================
            // Random direction generation (equal-area projection of sphere onto rectangle)  : https://math.stackexchange.com/questions/44689/how-to-find-a-random-axis-or-unit-vector-in-3d
            // =============================================================================================================================================================================
            float2 uniformRandoms = float2{ rnd(seed), rnd(seed) };
            float randomTheta = uniformRandoms.x * 2 * PI;
            float randomZ = (uniformRandoms.y * 2.0f) - 1.0f;
            float3 randomDir = float3{ sqrtf(1 - (randomZ * randomZ)) * cos(randomTheta), sqrtf(1 - (randomZ * randomZ)) * sin(randomTheta), randomZ };

            // If the generated random direction is not in the oriented hemisphere, invert it
            if (dot(randomDir, uvNormal3f) < 0)
            {
                randomDir = float3{-randomDir.x, -randomDir.y, -randomDir.z};
            }

            //// =================================================================================================================================================================================
            //// Random direction generation (uniform direction generation with spherical coords)  : https://math.stackexchange.com/questions/44689/how-to-find-a-random-axis-or-unit-vector-in-3d
            //// =================================================================================================================================================================================
            //float2 uniformRandoms = float2{ rnd(seed), rnd(seed) };
            //float theta = acos(1.0f - (2.0f * uniformRandoms.x));
            //float phi = 2 * PI * uniformRandoms.y;
            //float3 randomDir = float3{ sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta) };
            //
            //// If the generated random direction is not in the oriented hemisphere, invert it
            //if (dot(normalize(randomDir), normalize(uvNormal3f)) < 0)
            //{
            //    randomDir = float3{-randomDir.x, -randomDir.y, -randomDir.z};
            //}

            RadianceCellScatterUnbiasedPRD prd;
            unsigned int u0, u1, u2;
            // Trace ray against scene geometry to see if ray is occluded
            optixTrace(optixLaunchParams.sceneTraversable,
                rayOrigin3f,
                randomDir,
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

            prd.resultColor = glm::vec3{__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2)};

            // Cosine weighted contribution
            float cosContribution = dot(normalize(randomDir), normalize(uvNormal3f));
            totalRadiance += glm::vec3{ cosContribution * prd.resultColor.x, cosContribution * prd.resultColor.y, cosContribution * prd.resultColor.z };
            ++numSamples;
        }

        // "Diffuse BRDF" 
        totalRadiance *= diffuseColor;

        // Monte-Carlo weighted estimation
        const float r_result = totalRadiance.x / (float(numSamples) * 2 * PI);
        const float g_result = totalRadiance.y / (float(numSamples) * 2 * PI);
        const float b_result = totalRadiance.z / (float(numSamples) * 2 * PI);
        
        float4 resultValue = float4{ r_result, g_result, b_result, 0.0f };
        surf2Dwrite(resultValue, optixLaunchParams.currentBounceTextures[gameObjectNr], int(uv.x * optixLaunchParams.objectTextureResolutions[gameObjectNr]) * 16, int(uv.y * optixLaunchParams.objectTextureResolutions[gameObjectNr]));
    }
}