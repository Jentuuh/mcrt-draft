#include "renderer.hpp"

// This include may only appear in a SINGLE src file:
#include <optix_function_table_definition.h>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/string_cast.hpp>

// std
#include <iostream>

namespace mcrt {
    extern "C" char embedded_ptx_code[];

    // SBT record for a raygen program
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        void* data;
    };

    // SBT record for a miss program
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        void* data;
    };

    // SBT record for a hitgroup program
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        int objectID;
    };

    void TriangleMesh::addCube(const glm::vec3& center, const glm::vec3& size)
    {
        glm::mat4x4 xfm = glm::mat4x4(1.0f);
        xfm = glm::column(xfm, 3, glm::vec4{ center - 0.5f * size, 1.0f });
        xfm = glm::column(xfm, 0, glm::vec4{ size.x, 0.f, 0.f, 0.0f });
        xfm = glm::column(xfm, 1, glm::vec4{ 0.0f, size.y, 0.f, 0.0f });
        xfm = glm::column(xfm, 2, glm::vec4{ 0.0f, 0.0f, size.z, 0.0f });

        addUnitCube(xfm);
    }

    void TriangleMesh::addUnitCube(const glm::mat4x4& xfm)
    {
        int firstVertexID = (int)vertex.size();
        vertex.push_back(xfm * glm::vec4{ 0.0f, 0.0f, 0.0f, 1.0f });
        vertex.push_back(xfm * glm::vec4{ 1.0f, 0.0f, 0.0f, 1.0f });
        vertex.push_back(xfm * glm::vec4{ 0.0f, 1.0f, 0.0f, 1.0f });
        vertex.push_back(xfm * glm::vec4{ 1.0f, 1.0f, 0.0f, 1.0f });
        vertex.push_back(xfm * glm::vec4{ 0.0f, 0.0f, 1.0f, 1.0f });
        vertex.push_back(xfm * glm::vec4{ 1.0f, 0.0f, 1.0f, 1.0f });
        vertex.push_back(xfm * glm::vec4{ 0.0f, 1.0f, 1.0f, 1.0f });
        vertex.push_back(xfm * glm::vec4{ 1.0f, 1.0f, 1.0f, 1.0f });

        int indices[] = { 0,1,3, 2,3,0,
                         5,7,6, 5,6,4,
                         0,4,5, 0,5,1,
                         2,3,7, 2,7,6,
                         1,5,7, 1,7,3,
                         4,0,2, 4,2,6
                        };

        for (int i = 0; i < 12; i++)
            index.push_back(firstVertexID + glm::ivec3(indices[3 * i + 0],
                                                        indices[3 * i + 1],
                                                        indices[3 * i + 2]));
    }

    Renderer::Renderer(Scene& model, const Camera& camera): renderCamera{camera}
    {
        initOptix();
        updateCamera(camera);
        launchParams.frameID = 0;

        std::cout << "Creating OptiX context..." << std::endl;
        createContext();

        std::cout << "Setting up module..." << std::endl;
        createModule();

        std::cout << "Creating raygen programs..." << std::endl;
        createRaygenPrograms();
        std::cout << "Creating miss programs..." << std::endl;
        createMissPrograms();
        std::cout << "Creating hitgroup programs..." << std::endl;
        createHitGroupPrograms();

        launchParams.traversable = buildAccel(model);

        std::cout << "Setting up OptiX pipeline..." << std::endl;
        createPipeline();

        std::cout << "Building SBT..." << std::endl;
        buildSBT();

        // Allocate device space for launch parameters
        launchParamsBuffer.alloc(sizeof(launchParams));
        std::cout << "Context, module, pipeline, etc, all set up." << std::endl;

        std::cout << "MCRT renderer fully set up." << std::endl;
    }

    OptixTraversableHandle Renderer::buildAccel(Scene& model)
    {
        // Upload model to the device: the builder
        vertexBuffer.alloc_and_upload(model.vertices());
        indexBuffer.alloc_and_upload(model.indices());

        OptixTraversableHandle asHandle{ 0 };

        // ==================================================================
        // Triangle inputs
        // ==================================================================
        OptixBuildInput triangleInput = {};
        triangleInput.type
            = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        // Create local variables, because we need a *pointer* to the
        // device pointers
        CUdeviceptr d_vertices = vertexBuffer.d_pointer();
        CUdeviceptr d_indices = indexBuffer.d_pointer();

        triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput.triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
        triangleInput.triangleArray.numVertices = (int)model.vertices().size();
        triangleInput.triangleArray.vertexBuffers = &d_vertices;

        triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput.triangleArray.indexStrideInBytes = sizeof(glm::ivec3);
        triangleInput.triangleArray.numIndexTriplets = (int)model.indices().size();
        triangleInput.triangleArray.indexBuffer = d_indices;

        uint32_t triangleInputFlags[1] = { 0 };

        // In this example we have one SBT entry, and no per-primitive
        // materials:
        triangleInput.triangleArray.flags = triangleInputFlags;
        triangleInput.triangleArray.numSbtRecords = 1;
        triangleInput.triangleArray.sbtIndexOffsetBuffer = 0;
        triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
        triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

        // ==================================================================
        // BLAS setup
        // ==================================================================

        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
            | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
            ;
        accelOptions.motionOptions.numKeys = 1;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes blasBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage
        (optixContext,
            &accelOptions,
            &triangleInput,
            1,  // num_build_inputs
            &blasBufferSizes
        ));

        // ==================================================================
        // Prepare compaction
        // ==================================================================

        CUDABuffer compactedSizeBuffer;
        compactedSizeBuffer.alloc(sizeof(uint64_t));

        OptixAccelEmitDesc emitDesc;
        emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitDesc.result = compactedSizeBuffer.d_pointer();

        // ==================================================================
        // Execute build (main stage)
        // ==================================================================
       
        // Remember from pbrt that we can calculate the max. size of an AS tree beforehand
        CUDABuffer tempBuffer;
        tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

        CUDABuffer outputBuffer;
        outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);


        OPTIX_CHECK(optixAccelBuild(optixContext,
            /* stream */0,
            &accelOptions,
            &triangleInput,
            1,
            tempBuffer.d_pointer(),
            tempBuffer.sizeInBytes,

            outputBuffer.d_pointer(),
            outputBuffer.sizeInBytes,

            &asHandle,

            &emitDesc, 1
        ));
        CUDA_SYNC_CHECK();

        // ==================================================================
        // perform compaction
        // ==================================================================
        uint64_t compactedSize;
        compactedSizeBuffer.download(&compactedSize, 1);

        accelerationStructBuffer.alloc(compactedSize);
        OPTIX_CHECK(optixAccelCompact(optixContext,
            /*stream:*/0,
            asHandle,
            accelerationStructBuffer.d_pointer(),
            accelerationStructBuffer.sizeInBytes,
            &asHandle));
        CUDA_SYNC_CHECK();

        // ==================================================================
        // aaaaaand .... clean up
        // ==================================================================
        outputBuffer.free(); // << the UNcompacted, temporary output buffer
        tempBuffer.free();
        compactedSizeBuffer.free();

        return asHandle;
    }


    void Renderer::initOptix()
    {
        // Get CUDA compatible devices
        cudaFree(0);
        int numDevices;
        cudaGetDeviceCount(&numDevices);
        if (numDevices == 0)
            throw std::runtime_error("No CUDA capable devices found!");
        std::cout << "Found " << numDevices << " CUDA devices" << std::endl;

        // Initialize OptiX
        OPTIX_CHECK(optixInit());
        std::cout << "Successfully initialized OptiX. Hooray!" << std::endl;
    }

    // Logging callback for device context in case there is an error
    static void context_log_cb(unsigned int level,
                               const char* tag,
                               const char* message,
                               void*)
    {
        fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
    }

    // Creates and configures OptiX device context (for now only for the primary GPU)
    void Renderer::createContext()
    {
        const int deviceID = 0;
        CUDA_CHECK(SetDevice(deviceID));
        CUDA_CHECK(StreamCreate(&stream));

        cudaGetDeviceProperties(&deviceProperties, deviceID);
        std::cout << "Running on device: " << deviceProperties.name << std::endl;

        CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
        if (cuRes != CUDA_SUCCESS)
            fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

        OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
        OPTIX_CHECK(optixDeviceContextSetLogCallback
        (optixContext, context_log_cb, nullptr, 4));
    }

    // Creates module which contains all device programs.
    // For now we have a single module from a single .cu file, 
    // using a single embedded ptx string.
    void Renderer::createModule()
    {
        moduleCompileOptions.maxRegisterCount = 50;
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        pipelineCompileOptions = {};
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipelineCompileOptions.usesMotionBlur = false;
        pipelineCompileOptions.numPayloadValues = 2;
        pipelineCompileOptions.numAttributeValues = 2;
        pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

        // Max # of ray bounces
        pipelineLinkOptions.maxTraceDepth = 2;

        const std::string ptxCode = embedded_ptx_code;

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
            &moduleCompileOptions,
            &pipelineCompileOptions,
            ptxCode.c_str(),
            ptxCode.size(),
            log, &sizeof_log,
            &module
        ));
        if (sizeof_log > 1)
        {
            std::cout << log << std::endl;
        }
    }

    // Setup for raygen program(s)
    void Renderer::createRaygenPrograms()
    {
        // Single ray gen program for now
        raygenPGs.resize(1);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pgDesc.raygen.module = module;
        pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

        // OptixProgramGroup raypg;
        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optixContext,
            &pgDesc,
            1,
            &pgOptions,
            log, &sizeof_log,
            &raygenPGs[0]
        ));

        if (sizeof_log > 1)
        {
            std::cout << log << std::endl;
        }
    }

    // Setup for miss program(s)
    void Renderer::createMissPrograms()
    {        
        // Single miss program for now
        missPGs.resize(1);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgDesc.miss.module = module;
        pgDesc.miss.entryFunctionName = "__miss__radiance";

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optixContext,
            &pgDesc,
            1,
            &pgOptions,
            log, &sizeof_log,
            &missPGs[0]
        ));

        if (sizeof_log > 1)
        {
            std::cout << log << std::endl;
        }
    }

    // Setup for hitgroup program(s)
    void Renderer::createHitGroupPrograms()
    {
        // Single hitgroup program for now
        hitgroupPGs.resize(1);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pgDesc.hitgroup.moduleCH = module;
        pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        pgDesc.hitgroup.moduleAH = module;
        pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
    
        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optixContext,
            &pgDesc,
            1,
            &pgOptions,
            log, &sizeof_log,
            &hitgroupPGs[0]
        ));

        if (sizeof_log > 1)
        {
            std::cout << log << std::endl;
        }
    }

    // Assembles full pipeline of all programs
    void Renderer::createPipeline()
    {
        // Gather all program groups
        std::vector<OptixProgramGroup> programGroups;
        for (auto pg : raygenPGs)
            programGroups.push_back(pg);
        for (auto pg : missPGs)
            programGroups.push_back(pg);
        for (auto pg : hitgroupPGs)
            programGroups.push_back(pg);

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixPipelineCreate(optixContext,
            &pipelineCompileOptions,
            &pipelineLinkOptions,
            programGroups.data(),
            (int)programGroups.size(),
            log, &sizeof_log,
            &pipeline
        ));

        if (sizeof_log > 1)
        {
            std::cout << log << std::endl;
        }

        OPTIX_CHECK(optixPipelineSetStackSize
        (/* [in] The pipeline to configure the stack size for */
            pipeline,
            /* [in] The direct stack size requirement for direct
               callables invoked from IS or AH. */
            2 * 1024,
            /* [in] The direct stack size requirement for direct
               callables invoked from RG, MS, or CH.  */
            2 * 1024,
            /* [in] The continuation stack requirement. */
            2 * 1024,
            /* [in] The maximum depth of a traversable graph
               passed to trace. */
            1))
    }


    // Construct shader binding table
    void Renderer::buildSBT()
    {
        // ----------------------------------------
        // Build raygen records
        // ----------------------------------------
        std::vector<RaygenRecord> raygenRecords;
        for (int i = 0; i < raygenPGs.size(); i++) {
            RaygenRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
            rec.data = nullptr; /* for now ... */
            raygenRecords.push_back(rec);
        }
        // Upload records to device
        raygenRecordsBuffer.alloc_and_upload(raygenRecords);
        // Maintain a pointer to the device memory
        sbt.raygenRecord = raygenRecordsBuffer.d_pointer();


        // ----------------------------------------
        // Build miss records
        // ----------------------------------------
        std::vector<MissRecord> missRecords;
        for (int i = 0; i < missPGs.size(); i++) {
            MissRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
            rec.data = nullptr; /* for now ... */
            missRecords.push_back(rec);
        }
        // Upload records to device
        missRecordsBuffer.alloc_and_upload(missRecords);
        // Maintain a pointer to the device memory
        sbt.missRecordBase = missRecordsBuffer.d_pointer();
        
        sbt.missRecordStrideInBytes = sizeof(MissRecord);
        sbt.missRecordCount = (int)missRecords.size();


        // ----------------------------------------
        // Build hitgroup records
        // ----------------------------------------
        // TODO: FOR NOW THIS IS JUST A DUMMY VARIABLE, CHANGE THIS!!!
        int numObjects = 1;
        std::vector<HitgroupRecord> hitgroupRecords;
        for (int i = 0; i < numObjects; i++) {
            int objectType = 0;
            HitgroupRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[objectType], &rec));
            rec.objectID = i;
            hitgroupRecords.push_back(rec);
        }
        // Upload records to device
        hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
        // Maintain a pointer to the device memory
        sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
        sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
        sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
    }

    // Render loop
    void Renderer::render()
    {
        // First resize needs to be done before rendering
        if (launchParams.frame.size.x == 0) return;

        launchParamsBuffer.upload(&launchParams, 1);

        // Launch render pipeline
        OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
            pipeline, stream,
            /*! launch parameters and SBT */
            launchParamsBuffer.d_pointer(),
            launchParamsBuffer.sizeInBytes,
            &sbt,
            /*! dimensions of the launch: */
            launchParams.frame.size.x,
            launchParams.frame.size.y,
            1
        ));

        launchParams.frameID++;

        // TODO: implement double buffering!!!
        // sync - make sure the frame is rendered before we download and
        // display (obviously, for a high-performance application you
        // want to use streams and double-buffering, but for this simple
        // example, this will have to do)
        CUDA_SYNC_CHECK();
    }

    void Renderer::updateCamera(const Camera& camera)
    {
        renderCamera = camera;
        launchParams.camera.position = camera.position;
        launchParams.camera.direction = normalize(camera.target - camera.position);
        const float cosFovy = 0.66f;
        const float aspect = launchParams.frame.size.x / float(launchParams.frame.size.y);
        launchParams.camera.horizontal
            = cosFovy * aspect * normalize(cross(launchParams.camera.direction,
                camera.up));
        launchParams.camera.vertical
            = cosFovy * normalize(cross(launchParams.camera.horizontal,
                launchParams.camera.direction));
    }

    void Renderer::resize(const glm::ivec2& newSize)
    {
        // If window minimized
        if (newSize.x == 0 | newSize.y == 0) return;

        // Resize CUDA frame buffer
        colorBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));
    
        // Update launch parameters that are passed to OptiX launch
        launchParams.frame.size = newSize;
        launchParams.frame.colorBuffer = (uint32_t*)colorBuffer.d_pointer();

        // Reset camera, aspect may have changed
        updateCamera(renderCamera);
    }

    // Copy rendered color buffer from device to host memory for display
    void Renderer::downloadPixels(uint32_t h_pixels[])
    {
        colorBuffer.download(h_pixels,
            launchParams.frame.size.x * launchParams.frame.size.y);
    }
}