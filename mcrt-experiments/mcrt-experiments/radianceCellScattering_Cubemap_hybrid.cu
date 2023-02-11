#pragma once

#include <optix_device.h>
#include "random.hpp"
#include "vec_math.hpp"

#include "LaunchParams.hpp"
#include "glm/glm.hpp"

#include "cube_mapping.cuh"
#include "utils.cuh"

#define PI 3.14159265358979323846f
#define NUM_SAMPLES_HEMISPHERE 400
#define TRACING_RANGE 0.1f

using namespace mcrt;


namespace mcrt {
    extern "C" __constant__ LaunchParamsRadianceCellScatterCubeMap optixLaunchParams;

    static __forceinline__ __device__ RadianceCellScatterPRDHybrid loadRadianceCellScatterPRD()
    {
        RadianceCellScatterPRDHybrid prd = {};

        prd.hitFound = optixGetPayload_0();
        prd.resultColor.x = __uint_as_float(optixGetPayload_1());
        prd.resultColor.y = __uint_as_float(optixGetPayload_2());
        prd.resultColor.z = __uint_as_float(optixGetPayload_3());

        return prd;
    }

    static __forceinline__ __device__ void storeRadianceCellScatterPRD(RadianceCellScatterPRDHybrid prd)
    {
        optixSetPayload_0(prd.hitFound);
        optixSetPayload_1(__float_as_uint(prd.resultColor.x));
        optixSetPayload_2(__float_as_uint(prd.resultColor.y));
        optixSetPayload_3(__float_as_uint(prd.resultColor.z));
    }


    extern "C" __global__ void __closesthit__radiance__cell__scattering__scene()
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

        const int uTexelCoord = tc.x * optixLaunchParams.prevBounceTexture.size;
        const int vTexelCoord = tc.y * optixLaunchParams.prevBounceTexture.size;

        // Read color (outgoing radiance) at intersection (NOTE THAT WE ASSUME LAMBERTIAN SURFACE HERE)
        // --> Otherwise BRDF needs to be evaluated for the incoming direction at this point
        uint32_t lightSrcColor = optixLaunchParams.prevBounceTexture.colorBuffer[vTexelCoord * optixLaunchParams.prevBounceTexture.size + uTexelCoord];

        // Extract rgb values from light source texture pixel
        uint32_t r = 0x000000ff & (lightSrcColor);
        uint32_t g = (0x0000ff00 & (lightSrcColor)) >> 8;
        uint32_t b = (0x00ff0000 & (lightSrcColor)) >> 16;

        RadianceCellScatterPRDHybrid prd = loadRadianceCellScatterPRD();
        prd.hitFound = 1;
        prd.resultColor = glm::vec3{ r,g,b };
        storeRadianceCellScatterPRD(prd);
    }

    extern "C" __global__ void __anyhit__radiance__cell__scattering__scene() {
        // Do nothing
    }

    extern "C" __global__ void __miss__radiance__cell__scattering()
    {
        // Do nothing
    }

    extern "C" __global__ void __raygen__renderFrame__cell__scattering()
    {
        const int uvIndex = optixGetLaunchIndex().x;
        const int nonEmptyCellIndex = optixLaunchParams.nonEmptyCellIndex;
        const glm::ivec3 cellCoords = optixLaunchParams.cellCoords;
        const int probeResWidth = optixLaunchParams.probeWidthRes;
        const int probeResHeight = optixLaunchParams.probeHeightRes;

        // Take different seed for each radiance cell face
        unsigned int seed = tea<4>(uvIndex, nonEmptyCellIndex);

        // Get UV world position for this shader pass
        const int uvInsideOffset = optixLaunchParams.uvsInsideOffsets[nonEmptyCellIndex];
        glm::vec2 uv = optixLaunchParams.uvsInside[uvInsideOffset + uvIndex];
        const int u = int(uv.x * optixLaunchParams.uvWorldPositions.size);
        const int v = int(uv.y * optixLaunchParams.uvWorldPositions.size);

        glm::vec3 UVWorldPos = optixLaunchParams.uvWorldPositions.UVDataBuffer[v * optixLaunchParams.uvWorldPositions.size + u].worldPosition;
        const glm::vec3 UVNormal = optixLaunchParams.uvWorldPositions.UVDataBuffer[v * optixLaunchParams.uvWorldPositions.size + u].worldNormal;
        const glm::vec3 diffuseColor = optixLaunchParams.uvWorldPositions.UVDataBuffer[v * optixLaunchParams.uvWorldPositions.size + u].diffuseColor;

        float3 uvNormal3f = float3{ UVNormal.x, UVNormal.y, UVNormal.z };

        // We apply a small offset of 0.00001f in the direction of the normal to the UV world pos, to 'mitigate' floating point rounding errors causing false occlusions/illuminations
        UVWorldPos = glm::vec3{ UVWorldPos.x + UVNormal.x * 0.0001f, UVWorldPos.y + UVNormal.y * 0.0001f, UVWorldPos.z + UVNormal.z * 0.0001f };

        // Convert to float3 format
        float3 rayOrigin3f = float3{ UVWorldPos.x, UVWorldPos.y, UVWorldPos.z };

        // Center of this radiance cell
        glm::vec3 cellCenter = optixLaunchParams.cellCenter;
        
        // Probe buffer offset
        int probeOffset = ((cellCoords.z * probeResWidth * probeResHeight) + (cellCoords.y * probeResWidth) + cellCoords.x) * 6 * (optixLaunchParams.cubeMapResolution * optixLaunchParams.cubeMapResolution);

        // Radiance accumulator
        glm::vec3 totalRadiance = glm::vec3{ 0.0f, 0.0f, 0.0f };

        // Number of samples accumulator
        int numSamples = 0;

        // =============================================
        // Take hemisphere samples of incoming radiance
        // =============================================
        for (int s = 0; s < NUM_SAMPLES_HEMISPHERE; s++)
        {
            // Generate random direction on sphere
            float2 uniformRandoms = float2{ rnd(seed), rnd(seed) };
            float theta = acos(1.0f - (2.0f * uniformRandoms.x));
            float phi = 2 * PI * uniformRandoms.y;
            float3 randomDir = float3{ sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta) };
            
            // If the generated random direction is not in the oriented hemisphere, invert it
            if (dot(normalize(randomDir), normalize(uvNormal3f)) < 0)
            {
                randomDir = float3{-randomDir.x, -randomDir.y, -randomDir.z};
            }

            // Note that this is important! A ray has the following equation: `O + td`. If d is normalized, 
            // t represents the exact range or length along which we trace the ray. This assumption is necessary
            // when we want to set a maximum tracing range.
            randomDir = normalize(randomDir);

            // ===============================================
            // Test ray for intersections within tracing range
            // ===============================================
            RadianceCellScatterPRDHybrid prd{};
            prd.hitFound = 0;

            unsigned int u0, u1, u2, u3;
            u0 = prd.hitFound;

            // Trace ray. In case we find an intersection within the tracing range, we let the indirect light source contribute.
            // In case the ray hits nothing within the tracing range, we approximate "distant lighting" using the most nearby light probe.
            optixTrace(optixLaunchParams.sceneTraversable,
                rayOrigin3f,
                randomDir,
                0.f,            // tmin
                TRACING_RANGE,  // tmax (tracing range)
                0.0f,           // rayTime
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,     
                0,  // SBT offset
                1,  // SBT stride
                0,  // missSBTIndex
                u0, u1, u2, u3
            );

            prd.resultColor = glm::vec3{ __uint_as_float(u1), __uint_as_float(u2), __uint_as_float(u3) };

            if(u0 == 1)  // TRACED CONTRIBUTION
            {
                // Cosine weighted contribution
                float cosContribution = dot(normalize(randomDir), normalize(uvNormal3f));
                totalRadiance += glm::vec3{ cosContribution * prd.resultColor.x, cosContribution * prd.resultColor.y, cosContribution * prd.resultColor.z };
                ++numSamples;
            }
            else {  // PROBE CONTRIBUTION (How to make the distinction between an actual miss and out of range?)
                // Find "distant projection" along ray direction of point that we are calculating incoming radiance for, 
                // this is necessary to sample an approximated correct direction on the radiance probes.
                glm::vec3 distantProjectedPoint;
                find_distant_point_along_direction(UVWorldPos, glm::vec3{ randomDir.x, randomDir.y, randomDir.z }, &distantProjectedPoint);
                //printf("UVWorldPos:%f %f %f , projected: %f %f %f, dir: %f %f %f \n", UVWorldPos.x, UVWorldPos.y, UVWorldPos.z, distantProjectedPoint.x, distantProjectedPoint.y, distantProjectedPoint.z, randomDir.x, randomDir.y, randomDir.z);

                float faceU, faceV;
                int cubeMapFaceIndex;

                // ==================================================================================
                // Sample the probe in the center of this cell
                // ==================================================================================
                glm::vec3 probeSampleDirection = distantProjectedPoint - cellCenter;
                convert_xyz_to_cube_uv(probeSampleDirection.x, probeSampleDirection.y, probeSampleDirection.z, &cubeMapFaceIndex, &faceU, &faceV);

                int uIndex = optixLaunchParams.cubeMapResolution * faceU;
                int vIndex = optixLaunchParams.cubeMapResolution * faceV;
                int uvOffset = vIndex * optixLaunchParams.cubeMapResolution + uIndex;

                uint32_t incomingRadiance = optixLaunchParams.cubeMaps[(probeOffset + cubeMapFaceIndex * (optixLaunchParams.cubeMapResolution * optixLaunchParams.cubeMapResolution)) + uvOffset];

                // Extract rgb values from light source texture pixel
                uint32_t r = 0x000000ff & (incomingRadiance);
                uint32_t g = (0x0000ff00 & (incomingRadiance)) >> 8;
                uint32_t b = (0x00ff0000 & (incomingRadiance)) >> 16;
                glm::vec3 rgbNormalizedSpectrum = glm::vec3{ r, g, b };

                //// Convert to grayscale (for now we assume 1 color channel)
                //const float intensity = (0.3 * r + 0.59 * g + 0.11 * b) / 255.0f;

                // Cosine weighted contribution
                float cosContribution = dot(normalize(randomDir), normalize(uvNormal3f));
                if (cosContribution >= 0)
                {
                    totalRadiance += cosContribution * rgbNormalizedSpectrum;
                    numSamples++;
                }
            }
        }

        const int r = int(totalRadiance.x / float(numSamples));
        const int g = int(totalRadiance.y / float(numSamples));
        const int b = int(totalRadiance.z / float(numSamples));

        // convert to 32-bit rgba value (we explicitly set alpha to 0xff
        // to make stb_image_write happy ...
        const uint32_t rgba = 0xff000000
            | (r << 0) | (g << 8) | (b << 16);

        optixLaunchParams.currentBounceTexture.colorBuffer[v * optixLaunchParams.currentBounceTexture.size + u] = rgba;
    }
}