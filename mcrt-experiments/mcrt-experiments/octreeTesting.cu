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
#define NUM_SAMPLES_PER_STRATIFY_CELL 5
#define STRATIFY_RES_X 5
#define STRATIFY_RES_Y 5

using namespace mcrt;


namespace mcrt {
    extern "C" __constant__ LaunchParamsRadianceCellScatterCubeMap optixLaunchParams;

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


    extern "C" __global__ void __closesthit__radiance__cell__scattering__scene()
    {
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

        float test1 = tex3D<float>(optixLaunchParams.octreeTexture, 0, 0, 0);
        float test2 = tex3D<float>(optixLaunchParams.octreeTexture, 1, 0, 0);
        float test3 = tex3D<float>(optixLaunchParams.octreeTexture, 2, 0, 0);

        printf("Texture value 1: %f \n", test1);
        printf("Texture value 2: %f \n", test2);
        printf("Texture value 3: %f \n", test3);

    }
}