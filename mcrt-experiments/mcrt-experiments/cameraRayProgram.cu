#pragma once

#include <optix_device.h>

#include "LaunchParams.hpp"
#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"
#include "glm/gtc/type_ptr.hpp"

using namespace mcrt;

namespace mcrt {
    
    /**
    * Final shader in the pipeline. Visualizes the camera rays by reading from the texture-baked illumination.
    */

    extern "C" __constant__ LaunchParamsTutorial optixLaunchParams;

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

    extern "C" __global__ void __closesthit__shadow()
    {
        /* not going to be used ... */
    }

    extern "C" __global__ void __closesthit__radiance()
    {
        const MeshSBTData& sbtData
            = *(const MeshSBTData*)optixGetSbtDataPointer();

        // ------------------------------------------------------------------
        // gather some basic hit information
        // ------------------------------------------------------------------
        const int   primID = optixGetPrimitiveIndex();
        const glm::ivec3 index = sbtData.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;


        // Barycentric tex coords
        const glm::vec2 tc
            = (1.f - u - v) * sbtData.texcoord[index.x]
            + u * sbtData.texcoord[index.y]
            + v * sbtData.texcoord[index.z];

        uint32_t colorDirect = optixLaunchParams.lightTexture.colorBuffer[int(tc.y * optixLaunchParams.lightTexture.size) * optixLaunchParams.lightTexture.size + int(tc.x * optixLaunchParams.lightTexture.size)];
        uint32_t colorSecondBounce = optixLaunchParams.lightTextureSecondBounce.colorBuffer[int(tc.y * optixLaunchParams.lightTextureSecondBounce.size) * optixLaunchParams.lightTextureSecondBounce.size + int(tc.x * optixLaunchParams.lightTextureSecondBounce.size)];
        uint32_t colorThirdBounce = optixLaunchParams.lightTextureThirdBounce.colorBuffer[int(tc.y * optixLaunchParams.lightTextureThirdBounce.size) * optixLaunchParams.lightTextureThirdBounce.size + int(tc.x * optixLaunchParams.lightTextureThirdBounce.size)];
        
        // Extract rgb values from light source texture pixel
        const uint32_t r_direct = 0x000000ff & (colorDirect);
        const uint32_t r_second = 0x000000ff & (colorSecondBounce);
        const uint32_t r_third = 0x000000ff & (colorThirdBounce);

        const uint32_t g_direct = (0x0000ff00 & (colorDirect)) >> 8;
        const uint32_t g_second = (0x0000ff00 & (colorSecondBounce)) >> 8;
        const uint32_t g_third = (0x0000ff00 & (colorThirdBounce)) >> 8;

        const uint32_t b_direct = (0x00ff0000 & (colorDirect)) >> 16;
        const uint32_t b_second = (0x00ff0000 & (colorSecondBounce)) >> 16;
        const uint32_t b_third = (0x00ff0000 & (colorThirdBounce)) >> 16;

        const uint32_t r_total = r_direct + r_second + r_third;
        const uint32_t g_total = g_direct + g_second + g_third;
        const uint32_t b_total = b_direct + b_second + b_third;

        const glm::vec3 diffuseColor_direct = { float(r_direct) / 255.0f, float(g_direct) / 255.0f, float(b_direct) / 255.0f };
        const glm::vec3 diffuseColor_second = { float(r_second) / 255.0f, float(g_second) / 255.0f, float(b_second) / 255.0f };
        const glm::vec3 diffuseColor_third = { float(r_third) / 255.0f, float(g_third) / 255.0f, float(b_third) / 255.0f };

        const glm::vec3 diffuseTotal = diffuseColor_direct + diffuseColor_second + diffuseColor_third;

        glm::vec3& prd = *(glm::vec3*)getPRD<glm::vec3>();
        prd = diffuseTotal;
    }

    extern "C" __global__ void __anyhit__radiance()
    { /*! for this simple example, this will remain empty */
    }
    extern "C" __global__ void __anyhit__shadow()
    { /*! not going to be used */
    }


    //------------------------------------------------------------------------------
    // miss program that gets called for any ray that did not have a
    // valid intersection
    //
    // as with the anyhit/closest hit programs, in this example we only
    // need to have _some_ dummy function to set up a valid SBT
    // ------------------------------------------------------------------------------

    extern "C" __global__ void __miss__radiance()
    {
        glm::vec3& prd = *(glm::vec3*)getPRD<glm::vec3>();
        prd = glm::vec3{ 1.0f, 1.0f, 1.0f };
    }

    extern "C" __global__ void __miss__shadow()
    {
        // we didn't hit anything, so the light is visible
        glm::vec3& prd = *(glm::vec3*)getPRD<glm::vec3>();
        prd = glm::vec3(1.f);
    }

    //------------------------------------------------------------------------------
    // ray gen program - the actual rendering happens in here
    //------------------------------------------------------------------------------
    extern "C" __global__ void __raygen__renderFrame()
    {
        // compute a test pattern based on pixel ID
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;

        const auto& camera = optixLaunchParams.camera;

        // our per-ray data for this example. what we initialize it to
        // won't matter, since this value will be overwritten by either
        // the miss or hit program, anyway
        glm::vec3 pixelColorPRD = glm::vec3(0.f);

        // the values we store the PRD pointer in:
        uint32_t u0, u1;
        packPointer(&pixelColorPRD, u0, u1);

        // normalized screen plane position, in [0,1]^2
        const glm::vec2 screen(glm::vec2{ float(ix) + .5f, float(iy) + .5f }
        / glm::vec2{ optixLaunchParams.frame.size });

        // generate ray direction
        glm::vec3 rayDir = glm::normalize(camera.direction
            + (screen.x - 0.5f) * camera.horizontal
            + (screen.y - 0.5f) * camera.vertical);

        float3 camPos = float3{ camera.position.x, camera.position.y, camera.position.z };
        float3 rayDirection = float3{ rayDir.x, rayDir.y, rayDir.z };

        optixTrace(optixLaunchParams.traversable,
            camPos,
            rayDirection,
            0.f,    // tmin
            1e20f,  // tmax
            0.0f,   // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
            RADIANCE_RAY_TYPE,            // SBT offset
            RAY_TYPE_COUNT,               // SBT stride
            RADIANCE_RAY_TYPE,            // missSBTIndex 
            u0, u1);

        const int r = int(255.99f * pixelColorPRD.x);
        const int g = int(255.99f * pixelColorPRD.y);
        const int b = int(255.99f * pixelColorPRD.z);

        // convert to 32-bit rgba value (we explicitly set alpha to 0xff
        // to make stb_image_write happy ...
        const uint32_t rgba = 0xff000000
            | (r << 0) | (g << 8) | (b << 16);


        // and write to frame buffer ...
        const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
        optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
    }


}