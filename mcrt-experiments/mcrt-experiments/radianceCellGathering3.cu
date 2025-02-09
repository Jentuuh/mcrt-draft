#pragma once

#include <optix_device.h>
#include "random.hpp"
#include "vec_math.hpp"

#include "LaunchParams.hpp"
#include "glm/glm.hpp"

#include "spherical_harmonics.cuh"

#define PI 3.14159265358979323846f
#define EPSILON 0.0000000000002f

using namespace mcrt;

namespace mcrt {

    extern "C" __constant__ LaunchParamsRadianceCellGather optixLaunchParams;


    static __forceinline__ __device__ RadianceCellGatherPRD loadRadianceCellGatherPRD()
    {
        RadianceCellGatherPRD prd = {};

        prd.distanceToClosestProxyIntersectionSquared = __uint_as_float(optixGetPayload_0());
        prd.rayOrigin.x = __uint_as_float(optixGetPayload_1());
        prd.rayOrigin.y = __uint_as_float(optixGetPayload_2());
        prd.rayOrigin.z = __uint_as_float(optixGetPayload_3());

        return prd;
    }

    static __forceinline__ __device__ void storeRadianceCellGatherPRD(RadianceCellGatherPRD prd)
    {
        optixSetPayload_0(__float_as_uint(prd.distanceToClosestProxyIntersectionSquared));
        optixSetPayload_1(__float_as_uint(prd.rayOrigin.x));
        optixSetPayload_2(__float_as_uint(prd.rayOrigin.y));
        optixSetPayload_3(__float_as_uint(prd.rayOrigin.z));
    }


    extern "C" __global__ void __closesthit__radiance__cell__gathering__scene()
    {
        const MeshSBTDataRadianceCellGather& sbtData
            = *(const MeshSBTDataRadianceCellGather*)optixGetSbtDataPointer();

        const int   primID = optixGetPrimitiveIndex();
        const glm::ivec3 index = sbtData.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        const glm::vec3 intersectionWorldPos =
            (1.f - u - v) * sbtData.vertex[index.x]
            + u * sbtData.vertex[index.y]
            + v * sbtData.vertex[index.z];

        RadianceCellGatherPRD prd = loadRadianceCellGatherPRD();
        float distanceToProxyIntersect = (((intersectionWorldPos.x - prd.rayOrigin.x) * (intersectionWorldPos.x - prd.rayOrigin.x)) + ((intersectionWorldPos.y - prd.rayOrigin.y) * (intersectionWorldPos.y - prd.rayOrigin.y)) + ((intersectionWorldPos.z - prd.rayOrigin.z) * (intersectionWorldPos.z - prd.rayOrigin.z)));

        prd.distanceToClosestProxyIntersectionSquared = distanceToProxyIntersect;
        storeRadianceCellGatherPRD(prd);
    }

    extern "C" __global__ void __anyhit__radiance__cell__gathering__scene() {
        // Do nothing
        printf("Hit scene!");
    }

    extern "C" __global__ void __miss__radiance__cell__gathering()
    {
    }


    extern "C" __global__ void __raygen__renderFrame__cell__gathering()
    {
        // Get thread indices
        const int nonEmptyCellIndex = optixLaunchParams.nonEmptyCellIndex;

        // Light source texture tiling
        const int tileX = optixGetLaunchIndex().x;
        const int tileY = optixGetLaunchIndex().y;

        const int tileSize = optixLaunchParams.lightSourceTexture.size / optixLaunchParams.divisionResolution;  // Should be a whole number!
        const int startU = tileX * tileSize;
        const int startV = tileY * tileSize;
        const int endU = startU + tileSize;
        const int endV = startV + tileSize;

        // Amount SH basis functions
        int amountBasisFunctions = optixLaunchParams.sphericalHarmonicsWeights.amountBasisFunctions;

        // Size of a radiance cell + dimensions of each stratified cell
        const float cellSize = optixLaunchParams.cellSize;

        // Accumulators (8*9, 8 corners, 9 basis functions per corner)
        float SHAccumulator[8 * 9] = { 0.0 };
        int numSamplesAccumulator[8] = { 0 };

        // Center of this radiance cell
        glm::vec3 cellCenter = optixLaunchParams.nonEmptyCells.centers[nonEmptyCellIndex];

        float3 ogSh0 = { cellCenter.x - 0.5f * cellSize, cellCenter.y - 0.5f * cellSize, cellCenter.z - 0.5f * cellSize };
        float3 ogSh1 = { cellCenter.x + 0.5f * cellSize, cellCenter.y - 0.5f * cellSize, cellCenter.z - 0.5f * cellSize };
        float3 ogSh2 = { cellCenter.x - 0.5f * cellSize, cellCenter.y + 0.5f * cellSize, cellCenter.z - 0.5f * cellSize };
        float3 ogSh3 = { cellCenter.x + 0.5f * cellSize, cellCenter.y + 0.5f * cellSize, cellCenter.z - 0.5f * cellSize };
        float3 ogSh4 = { cellCenter.x - 0.5f * cellSize, cellCenter.y - 0.5f * cellSize, cellCenter.z + 0.5f * cellSize };
        float3 ogSh5 = { cellCenter.x + 0.5f * cellSize, cellCenter.y - 0.5f * cellSize, cellCenter.z + 0.5f * cellSize };
        float3 ogSh6 = { cellCenter.x - 0.5f * cellSize, cellCenter.y + 0.5f * cellSize, cellCenter.z + 0.5f * cellSize };
        float3 ogSh7 = { cellCenter.x + 0.5f * cellSize, cellCenter.y + 0.5f * cellSize, cellCenter.z + 0.5f * cellSize };

        float3 shOrigins[8] = { ogSh0, ogSh1, ogSh2, ogSh3, ogSh4, ogSh5, ogSh6, ogSh7 };

        // Loop over cell's SHs
        for (int sh = 0; sh < 8; sh++)
        {
            // Loop over all texels of the light source texture tile
            for (int u = startU; u < endU; u++)
            {
                for (int v = startV; v < endV; v++)
                {
                    uint32_t lightSrcColor = optixLaunchParams.lightSourceTexture.colorBuffer[v * optixLaunchParams.lightSourceTexture.size + u];

                    // Extract rgb values from light source texture pixel
                    const uint32_t r = 0x000000ff & (lightSrcColor);
                    const uint32_t g = (0x0000ff00 & (lightSrcColor)) >> 8;
                    const uint32_t b = (0x00ff0000 & (lightSrcColor)) >> 16;

                    // Convert to grayscale (for now we assume 1 color channel)
                    const float grayscale = (0.3 * r + 0.59 * g + 0.11 * b) / 255.0f;

                    if (r > 0 && g > 0 && b > 0)
                    {
                        //printf("Non empty cell index:%d Cell center: %f %f %f\n", nonEmptyCellIndex, cellCenter.x, cellCenter.y, cellCenter.z);
                        //printf("Non empty cell index:%d UV:(%d,%d) tileX:%d tileY:%d colors: %d %d %d\n", nonEmptyCellIndex, u, v, tileX, tileY, r, g, b);
                    }

                    if (grayscale < 0.0001f)  // Skip pixels with no outgoing radiance
                        continue;

                    // World position + normal of the texel
                    glm::vec3 UVWorldPos = optixLaunchParams.uvWorldPositions.UVDataBuffer[v * optixLaunchParams.lightSourceTexture.size + u].worldPosition;
                    //printf("UV world pos: %f %f %f\n", UVWorldPos.x, UVWorldPos.y, UVWorldPos.z);

                    const glm::vec3 UVNormal = optixLaunchParams.uvWorldPositions.UVDataBuffer[v * optixLaunchParams.lightSourceTexture.size + u].worldNormal;
                    float3 uvNormal3f = float3{ UVNormal.x, UVNormal.y, UVNormal.z };

                    // We apply a small offset of 0.00001f in the direction of the normal to the UV world pos, to 'mitigate' floating point rounding errors causing false occlusions/illuminations
                    UVWorldPos = glm::vec3{ UVWorldPos.x + UVNormal.x * 0.0001f, UVWorldPos.y + UVNormal.y * 0.0001f, UVWorldPos.z + UVNormal.z * 0.0001f };
             
                    // Ray destination (SH origin)
                    glm::vec3 rayDestination = { shOrigins[sh].x, shOrigins[sh].y, shOrigins[sh].z };

                    // Ray direction
                    glm::vec3 rayDir = UVWorldPos - rayDestination;
                    glm::vec3 rayDirInv = rayDestination - UVWorldPos;

                    // Convert to float3 format
                    float3 rayOrigin3f = float3{ UVWorldPos.x, UVWorldPos.y, UVWorldPos.z };
                    float3 rayDir3f = float3{ rayDir.x, rayDir.y, rayDir.z };
                    float3 rayDir3fInv = float3{ rayDirInv.x, rayDirInv.y, rayDirInv.z };
                    float3 normalizedRayDir = normalize(rayDir3f);
                    float3 normalizedRayDirInv = normalize(rayDir3fInv);

                    // Check if the current SH is facing the light source, if not, skip it
                    if (dot(rayDir3fInv, uvNormal3f) < 0)
                        continue;


                    // Calculate spherical coordinate representation of ray
                    // (https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates)

                    //double theta = acos(clamp(normalizedRayDir.z, -1.0, 1.0));
                    //if (normalizedRayDir.x > 0)
                    //{
                    //    double phi = atan2(normalizedRayDir.y, normalizedRayDir.x); // atan2 returns something between -2*pi and 2*pi, but phi should be between [0;pi]
                    //}
                    //else if (normalizedRayDir.x < 0 && normalizedRayDir.y >= 0)
                    //{
                    //    double phi = atan2(normalizedRayDir.y, normalizedRayDir.x) + PI; 
                    //}
                    //else if (normalizedRayDir.x < 0 && normalizedRayDir.y < 0)
                    //{
                    //    double phi = atan2(normalizedRayDir.y, normalizedRayDir.x) - PI;
                    //}
                    //else if (normalizedRayDir.x == 0 && normalizedRayDir.y > 0)
                    //{
                    //    double phi = 0.5f * PI;
                    //}
                    //else {
                    //    double phi = -0.5f * PI;
                    //}

                    RadianceCellGatherPRD prd{};
                    prd.rayOrigin = UVWorldPos;

                    unsigned int u0, u1, u2, u3;

                    u0 = __float_as_uint(1000.0f); // Initialize distanceToProxySquared at 1000.0f, so in case the ray misses all geometry, the distance to the proxy is 'infinite'
                    u1 = __float_as_uint(prd.rayOrigin.x);
                    u2 = __float_as_uint(prd.rayOrigin.y);
                    u3 = __float_as_uint(prd.rayOrigin.z);

                    // Trace ray against scene geometry to see if ray is occluded
                    optixTrace(optixLaunchParams.sceneTraversable,
                        rayOrigin3f,
                        rayDir3fInv,
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

                    prd.distanceToClosestProxyIntersectionSquared = __uint_as_float(u0);

                    float distanceToSHSquared = (((rayDestination.x - UVWorldPos.x) * (rayDestination.x - UVWorldPos.x)) + ((rayDestination.y - UVWorldPos.y) * (rayDestination.y - UVWorldPos.y)) + ((rayDestination.z - UVWorldPos.z) * (rayDestination.z - UVWorldPos.z)));
                    //printf("distanceToProxy: %f distanceToSH: %f\n", prd.distanceToClosestProxyIntersectionSquared, distanceToSHSquared);

                    // No occlusion, we can let the ray contribute (visibility check)
                    if (distanceToSHSquared < prd.distanceToClosestProxyIntersectionSquared)
                    {
                        numSamplesAccumulator[sh]++;

                        // Ray contribution
                        SHAccumulator[sh * 9] += grayscale * Y_0_0();
                        SHAccumulator[sh * 9 + 1] += grayscale * Y_min1_1(normalizedRayDir.x, normalizedRayDir.y, normalizedRayDir.z);
                        SHAccumulator[sh * 9 + 2] += grayscale * Y_0_1(normalizedRayDir.x, normalizedRayDir.y, normalizedRayDir.z);
                        SHAccumulator[sh * 9 + 3] += grayscale * Y_1_1(normalizedRayDir.x, normalizedRayDir.y, normalizedRayDir.z);
                        SHAccumulator[sh * 9 + 4] += grayscale * Y_min2_2(normalizedRayDir.x, normalizedRayDir.y, normalizedRayDir.z);
                        SHAccumulator[sh * 9 + 5] += grayscale * Y_min1_2(normalizedRayDir.x, normalizedRayDir.y, normalizedRayDir.z);
                        SHAccumulator[sh * 9 + 6] += grayscale * Y_0_2(normalizedRayDir.x, normalizedRayDir.y, normalizedRayDir.z);
                        SHAccumulator[sh * 9 + 7] += grayscale * Y_1_2(normalizedRayDir.x, normalizedRayDir.y, normalizedRayDir.z);
                        SHAccumulator[sh * 9 + 8] += grayscale * Y_2_2(normalizedRayDir.x, normalizedRayDir.y, normalizedRayDir.z);
                    }
                }
            }
        }

        // Current non-empty cell * amount of basis functions * 8 SHs per cell 
        int cellOffset = nonEmptyCellIndex * amountBasisFunctions * 8;

        // Write projected results to output buffer
        for (int n_samples_i = 0; n_samples_i < 8; n_samples_i++)
        {
            int numSamples = numSamplesAccumulator[n_samples_i];
            if (numSamples > 0) {
                atomicAdd(&optixLaunchParams.shNumSamplesAccumulators[nonEmptyCellIndex * 8 + n_samples_i], numSamples);
                //double weight = 1.0 / (numSamplesAccumulator[n_samples_i] * 4.0 * PI);
                for (int basis_f_i = 0; basis_f_i < 9; basis_f_i++)
                {
                    atomicAdd(&optixLaunchParams.sphericalHarmonicsWeights.weights[cellOffset + n_samples_i * amountBasisFunctions + basis_f_i], SHAccumulator[n_samples_i * amountBasisFunctions + basis_f_i]);

                    //// Accumulate
                    //if (!isnan(SHAccumulator[n_samples_i * 9 + basis_f_i]))
                    //{
                    //    atomicAdd(&optixLaunchParams.sphericalHarmonicsWeights.weights[cellOffset + n_samples_i * amountBasisFunctions + basis_f_i], SHAccumulator[n_samples_i * amountBasisFunctions + basis_f_i]);
                    //}
                    //optixLaunchParams.sphericalHarmonicsWeights.weights[cellOffset + n_samples_i * 9 + basis_f_i] += SHAccumulator[n_samples_i * 9 + basis_f_i];
                }
            }
        }
    }
}