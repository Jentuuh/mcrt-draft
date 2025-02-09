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

    extern "C" __constant__ LaunchParamsCameraPass optixLaunchParams;

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

        float4 directRadiance = tex2D<float4>(optixLaunchParams.directLightTextures[sbtData.objectNr], tc.x, tc.y);
        float4 secondBounceRadiance = tex2D<float4>(optixLaunchParams.secondBounceTextures[sbtData.objectNr], tc.x, tc.y);
        float4 thirdBounceRadiance = tex2D<float4>(optixLaunchParams.thirdBounceTextures[sbtData.objectNr], tc.x, tc.y);

        const glm::vec3 diffuseColor_direct = glm::vec3{ directRadiance.x, directRadiance.y, directRadiance.z };
        const glm::vec3 diffuseColor_second = glm::vec3{ secondBounceRadiance.x, secondBounceRadiance.y, secondBounceRadiance.z };
        const glm::vec3 diffuseColor_third = glm::vec3{ thirdBounceRadiance.x, thirdBounceRadiance.y, thirdBounceRadiance.z };

        const glm::vec3 diffuseTotal = diffuseColor_direct + diffuseColor_second + diffuseColor_third;

        // ==========================
        // HDR Reinhard Tone Mapping
        // ==========================
        const float gamma = 2.2f;
        // Reinhard
        glm::vec3 mapped = diffuseTotal / ((diffuseTotal + glm::vec3{1.0f, 1.0f, 1.0f}) * 2.0f);
        // Gamma correction 
        mapped = glm::vec3{ pow(mapped.x, 1.0f/ gamma), pow(mapped.y, 1.0f / gamma), pow(mapped.z, 1.0f / gamma) };

        // ==========================
        // HDR Exposure Tone Mapping
        // ==========================
        //const float exposure = 0.8f;
        //const float gamma = 2.2f;

        //glm::vec3 mapped = glm::vec3{ 1.0f, 1.0f, 1.0f } - glm::vec3{ exp(-diffuseTotal.x * exposure), exp(-diffuseTotal.y * exposure), exp(-diffuseTotal.z * exposure) };

        //// Gamma correction 
        //mapped = glm::vec3{ pow(mapped.x, 1.0f / gamma), pow(mapped.y, 1.0f / gamma), pow(mapped.z, 1.0f / gamma) };


        glm::vec3& prd = *(glm::vec3*)getPRD<glm::vec3>();
        prd = mapped;
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
        prd = glm::vec3{ 0.0f, 0.0f, 0.0f };
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

        const glm::vec2 d = 2.0f * glm::vec2{
            (static_cast<float>(ix)) / static_cast<float>(optixLaunchParams.frame.size.x),
            (static_cast<float>(iy)) / static_cast<float>(optixLaunchParams.frame.size.y)
        } - 1.0f;
        glm::vec3 rayDir = normalize(d.x * camera.horizontal + d.y * camera.vertical + camera.direction);

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