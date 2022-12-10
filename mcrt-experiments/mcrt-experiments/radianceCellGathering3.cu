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

        prd.distanceToClosestProxyIntersection = __uint_as_float(optixGetPayload_0());
        prd.rayOrigin.x = __uint_as_float(optixGetPayload_1());
        prd.rayOrigin.y = __uint_as_float(optixGetPayload_2());
        prd.rayOrigin.z = __uint_as_float(optixGetPayload_3());

        return prd;
    }

    static __forceinline__ __device__ void storeRadianceCellGatherPRD(RadianceCellGatherPRD prd)
    {
        optixSetPayload_0(__float_as_uint(prd.distanceToClosestProxyIntersection));
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

        prd.distanceToClosestProxyIntersection = distanceToProxyIntersect;
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
        const int nonEmptyCellIndex = optixGetLaunchIndex().x;

        // Light source texture tiling
        const int tileX = optixGetLaunchIndex().y;
        const int tileY = optixGetLaunchIndex().z;

        //printf("%d %d\n", tileX, tileX);

        const int tileNumber = tileY * optixLaunchParams.divisionResolution + tileX;    // Flattened linear tile number 
        const int tileSize = optixLaunchParams.lightSourceTexture.size / optixLaunchParams.divisionResolution;  // Should be a whole number!
        const int startU = tileX * tileSize;
        const int startV = tileY * tileSize;
        const int endU = startU + tileSize;
        const int endV = startV + tileSize;

        // Amount SH basis functions
        int amountBasisFunctions = optixLaunchParams.sphericalHarmonicsWeights.amountBasisFunctions;

        // Size of a radiance cell + dimensions of each stratified cell
        const float cellSize = optixLaunchParams.cellSize;

        //float stratifyCellWidth = cellSize / optixLaunchParams.stratifyResX;
        //float stratifyCellHeight = cellSize / optixLaunchParams.stratifyResY;

        //float stratifyCellWidthNormalized = 1.0 / optixLaunchParams.stratifyResX;
        //float stratifyCellHeightNormalized = 1.0 / optixLaunchParams.stratifyResY;

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

        float3 ogLeft{ cellCenter.x - 0.5f * cellSize, cellCenter.y - 0.5f * cellSize, cellCenter.z + 0.5f * cellSize };
        float3 ogRight{ cellCenter.x + 0.5f * cellSize, cellCenter.y - 0.5f * cellSize, cellCenter.z - 0.5f * cellSize };
        float3 ogUp{ cellCenter.x - 0.5f * cellSize, cellCenter.y + 0.5f * cellSize, cellCenter.z - 0.5f * cellSize };
        float3 ogDown{ cellCenter.x - 0.5f * cellSize, cellCenter.y - 0.5f * cellSize, cellCenter.z + 0.5f * cellSize };
        float3 ogFront{ cellCenter.x - 0.5f * cellSize, cellCenter.y - 0.5f * cellSize, cellCenter.z - 0.5f * cellSize };
        float3 ogBack{ cellCenter.x + 0.5f * cellSize, cellCenter.y - 0.5f * cellSize, cellCenter.z + 0.5f * cellSize };

        // LEFT, RIGHT, UP, DOWN, FRONT, BACK
        float3 cellNormals[6] = { float3{-1.0f, 0.0f, 0.0f}, float3{1.0f, 0.0f, 0.0f}, float3{0.0f, 1.0f, 0.0f}, float3{0.0f, -1.0f, 0.0f}, float3{0.0f, 0.0f, -1.0f}, float3{0.0f, 0.0f, 1.0f} };
        // Origin, du, dv for each face
        float3 faceOgDuDv[6][3] = { {ogLeft, float3{0.0f, 0.0f, -1.0f}, float3{0.0f, 1.0f, 0.0f} }, {ogRight, float3{0.0f, 0.0f, 1.0f},float3{0.0f, 1.0f, 0.0f} }, {ogUp, float3{1.0f, 0.0f, 0.0f},float3{0.0f, 0.0f, 1.0f} }, {ogDown, float3{1.0f, 0.0f, 0.0f},float3{0.0f, 0.0f, -1.0f}}, {ogFront, float3{1.0f, 0.0f, 0.0f},float3{0.0f, 1.0f, 0.0f} }, {ogBack, float3{-1.0f, 0.0f, 0.0f},float3{0.0f, 1.0f, 0.0f} } };
        // The indices of the SHs that belong to each face, to use while indexing the buffer (L,R,U,D,F,B), (LB, RB, LT, RT)
        int4 cellSHIndices[6] = { int4{4, 0, 6, 2}, int4{1, 5, 3, 7}, int4{2, 3, 6, 7}, int4{4, 5, 0, 1}, int4{0, 1, 2, 3}, int4{5, 4, 7, 6} };

        // Loop over cell's SHs
        for (int sh = 0; sh < 8; sh++)
        {
            // Loop over all texels of the light source texture
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

                    if (grayscale == 0.0f)  // Skip pixels with no outgoing radiance
                        continue;

                    // World position + normal of the texel
                    glm::vec3 UVWorldPos = optixLaunchParams.uvWorldPositions.UVDataBuffer[v * optixLaunchParams.lightSourceTexture.size + u].worldPosition;
                    const glm::vec3 UVNormal = optixLaunchParams.uvWorldPositions.UVDataBuffer[v * optixLaunchParams.lightSourceTexture.size + u].worldNormal;
                    float3 uvNormal3f = float3{ UVNormal.x, UVNormal.y, UVNormal.z };

                    // We apply a small offset of 0.00001f in the direction of the normal to the UV world pos, to 'mitigate' floating point rounding errors causing false occlusions/illuminations
                    UVWorldPos = glm::vec3{ UVWorldPos.x + UVNormal.x * 0.00001f, UVWorldPos.y + UVNormal.y * 0.00001f, UVWorldPos.z + UVNormal.z * 0.00001f };

                    glm::vec3 lightToCellDir = { cellCenter.x - UVWorldPos.x, cellCenter.y - UVWorldPos.y, cellCenter.z - UVWorldPos.z };

                    float3 rayOrigin3f = float3{ UVWorldPos.x, UVWorldPos.y, UVWorldPos.z };
                    float3 rayOgToCellCenter3f = float3{ lightToCellDir.x, lightToCellDir.y, lightToCellDir.z };

                    // Cosine between vector from ray origin to cell center and texel normal to check if cell is facing
                    double radCellFacing = dot(normalize(rayOgToCellCenter3f), uvNormal3f);

                    if (radCellFacing > 0)
                    {
                        // Ray destination (SH origin)
                        glm::vec3 rayDestination = { shOrigins[sh].x, shOrigins[sh].y, shOrigins[sh].z };

                        // Ray direction
                        glm::vec3 rayDir = UVWorldPos - rayDestination;

                        // Convert to float3 format
                        float3 rayOrigin3f = float3{ UVWorldPos.x, UVWorldPos.y, UVWorldPos.z };
                        float3 rayDir3f = float3{ rayDir.x, rayDir.y, rayDir.z };

                        // Calculate spherical coordinate representation of ray
                        // (https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates)
                        float3 normalizedRayDir = normalize(rayDir3f);
                        double theta = acos(normalizedRayDir.z);
                        int signY = signbit(normalizedRayDir.y) == 0 ? 1 : -1;
                        double phi = signY * acos(normalizedRayDir.x / (sqrtf((normalizedRayDir.x * normalizedRayDir.x) + (normalizedRayDir.y * normalizedRayDir.y))));

                        RadianceCellGatherPRD prd{};
                        prd.rayOrigin = UVWorldPos;

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

                        prd.distanceToClosestProxyIntersection = u0;
                        float distanceToGridIntersect = (((rayDestination.x - prd.rayOrigin.x) * (rayDestination.x - prd.rayOrigin.x)) + ((rayDestination.y - prd.rayOrigin.y) * (rayDestination.y - prd.rayOrigin.y)) + ((rayDestination.z - prd.rayOrigin.z) * (rayDestination.z - prd.rayOrigin.z)));

                        // No occlusion, we can let the ray contribute
                        if (distanceToGridIntersect < prd.distanceToClosestProxyIntersection)
                        {
                            numSamplesAccumulator[sh]++;

                            // Ray contribution
                            SHAccumulator[sh * 9] += grayscale * Y_0_0();
                            SHAccumulator[sh * 9 + 1] += grayscale * Y_min1_1(phi, theta);
                            SHAccumulator[sh * 9 + 2] += grayscale * Y_0_1(phi, theta);
                            SHAccumulator[sh * 9 + 3] += grayscale * Y_1_1(phi, theta);
                            SHAccumulator[sh * 9 + 4] += grayscale * Y_min2_2(phi, theta);
                            SHAccumulator[sh * 9 + 5] += grayscale * Y_min1_2(phi, theta);
                            SHAccumulator[sh * 9 + 6] += grayscale * Y_0_2(phi, theta);
                            SHAccumulator[sh * 9 + 7] += grayscale * Y_1_2(phi, theta);
                            SHAccumulator[sh * 9 + 8] += grayscale * Y_2_2(phi, theta);
                        }
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
            atomicAdd(&optixLaunchParams.shNumSamplesAccumulators[nonEmptyCellIndex * 8 + n_samples_i], numSamples);

            if (numSamples > 0) {
                //double weight = 1.0 / (numSamplesAccumulator[n_samples_i] * 4.0 * PI);
                for (int basis_f_i = 0; basis_f_i < 9; basis_f_i++)
                {
                    // Accumulate
                    if (!isnan(SHAccumulator[n_samples_i * 9 + basis_f_i]))
                    {
                        atomicAdd(&optixLaunchParams.sphericalHarmonicsWeights.weights[cellOffset + n_samples_i * amountBasisFunctions + basis_f_i], SHAccumulator[n_samples_i * amountBasisFunctions + basis_f_i]);
                    }
                    //optixLaunchParams.sphericalHarmonicsWeights.weights[cellOffset + n_samples_i * 9 + basis_f_i] += SHAccumulator[n_samples_i * 9 + basis_f_i];
                }
            }
        }
    }
}