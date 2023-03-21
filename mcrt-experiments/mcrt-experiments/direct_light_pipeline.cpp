#include "direct_light_pipeline.hpp"
#include <iostream>

namespace mcrt {
    extern "C" char embedded_ptx_code_direct_lighting_gathering[];

    // SBT record for a raygen program
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecordDirect
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        void* data;
    };

    // SBT record for a miss program
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecordDirect
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        void* data;
    };

    // SBT record for a hitgroup program
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecordDirect
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        MeshSBTDataDirectLighting data;
    };

	DirectLightPipeline::DirectLightPipeline(OptixDeviceContext& context, GeometryBufferHandle& geometryBuffers, Scene& scene) : McrtPipeline(context, geometryBuffers, scene)
	{
		init(context, geometryBuffers, scene);
        launchParams.traversable = buildAccelerationStructure(context, geometryBuffers, scene);
        launchParamsBuffer.alloc(sizeof(launchParams));
	}

    void DirectLightPipeline::uploadLaunchParams()
    {
        launchParamsBuffer.upload(&launchParams, 1);
    }


    void DirectLightPipeline::buildModule(OptixDeviceContext& context)
    {
        moduleCompileOptions.maxRegisterCount = 50;
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        pipelineCompileOptions = {};
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipelineCompileOptions.usesMotionBlur = false;
        pipelineCompileOptions.numPayloadValues = 9;
        pipelineCompileOptions.numAttributeValues = 2;
        pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

        // Max # of ray bounces
        pipelineLinkOptions.maxTraceDepth = 2;

        const std::string ptxCode = embedded_ptx_code_direct_lighting_gathering;

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixModuleCreateFromPTX(context,
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

    void DirectLightPipeline::buildDevicePrograms(OptixDeviceContext& context)
    {
        //---------------------------------------
        //  RAYGEN PROGRAMS
        //---------------------------------------
        raygenPGs.resize(1);

        OptixProgramGroupOptions pgOptionsRaygen = {};
        OptixProgramGroupDesc pgDescRaygen = {};
        pgDescRaygen.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pgDescRaygen.raygen.module = module;
        pgDescRaygen.raygen.entryFunctionName = "__raygen__renderFrame__direct__lighting";

        // OptixProgramGroup raypg;
        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(context,
            &pgDescRaygen,
            1,
            &pgOptionsRaygen,
            log, &sizeof_log,
            &raygenPGs[0]
        ));

        if (sizeof_log > 1)
        {
            std::cout << log << std::endl;
        }

        //---------------------------------------
        //  MISS PROGRAMS
        //---------------------------------------
        missPGs.resize(1);

        OptixProgramGroupOptions pgOptionsMiss = {};
        OptixProgramGroupDesc pgDescMiss = {};
        pgDescMiss.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgDescMiss.miss.module = module;

        // ------------------------------------------------------------------
        // radiance rays
        // ------------------------------------------------------------------
        pgDescMiss.miss.entryFunctionName = "__miss__radiance__direct__lighting";

        OPTIX_CHECK(optixProgramGroupCreate(context,
            &pgDescMiss,
            1,
            &pgOptionsMiss,
            log, &sizeof_log,
            &missPGs[0]
        ));

        if (sizeof_log > 1)
        {
            std::cout << log << std::endl;
        }

        //---------------------------------------
        //  HITGROUP PROGRAMS
        //---------------------------------------
         // Single hitgroup program for now
        hitgroupPGs.resize(1);

        OptixProgramGroupOptions pgOptionsHitgroup = {};
        OptixProgramGroupDesc    pgDescHitgroup = {};
        pgDescHitgroup.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pgDescHitgroup.hitgroup.moduleCH = module;
        pgDescHitgroup.hitgroup.moduleAH = module;

        pgDescHitgroup.hitgroup.entryFunctionNameCH = "__closesthit__radiance__direct__lighting";
        pgDescHitgroup.hitgroup.entryFunctionNameAH = "__anyhit__radiance__direct__lighting";

        OPTIX_CHECK(optixProgramGroupCreate(context,
            &pgDescHitgroup,
            1,
            &pgOptionsHitgroup,
            log, &sizeof_log,
            &hitgroupPGs[0]
        ));

        if (sizeof_log > 1)
        {
            std::cout << log << std::endl;
        }
    }

    void DirectLightPipeline::buildSBT(GeometryBufferHandle& geometryBuffers, Scene& scene)
    {
        // ----------------------------------------
        // Build raygen records
        // ----------------------------------------
        std::vector<RaygenRecordDirect> raygenRecords;
        for (int i = 0; i < raygenPGs.size(); i++) {
            RaygenRecordDirect rec;
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
        std::vector<MissRecordDirect> missRecords;
        for (int i = 0; i < missPGs.size(); i++) {
            MissRecordDirect rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
            rec.data = nullptr; /* for now ... */
            missRecords.push_back(rec);
        }
        // Upload records to device
        missRecordsBuffer.alloc_and_upload(missRecords);
        // Maintain a pointer to the device memory
        sbt.missRecordBase = missRecordsBuffer.d_pointer();

        sbt.missRecordStrideInBytes = sizeof(MissRecordDirect);
        sbt.missRecordCount = (int)missRecords.size();


        // ----------------------------------------
        // Build hitgroup records
        // ----------------------------------------
        int numObjects = scene.numObjects();
        std::vector<HitgroupRecordDirect> hitgroupRecords;
        for (int i = 0; i < numObjects; i++) {
            auto mesh = scene.getGameObjects()[i]->model->mesh;

            HitgroupRecordDirect rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0], &rec));

            rec.data.diffuseColor = mesh->diffuse;
            rec.data.vertex = (glm::vec3*)geometryBuffers.vertices[i].d_pointer();
            rec.data.index = (glm::ivec3*)geometryBuffers.indices[i].d_pointer();
            rec.data.normal = (glm::vec3*)geometryBuffers.normals[i].d_pointer();
            rec.data.texcoord = (glm::vec2*)geometryBuffers.texCoords[i].d_pointer();
            rec.data.diffuseUV = (glm::vec2*)geometryBuffers.diffuseUVs[i].d_pointer();

            hitgroupRecords.push_back(rec);  
        }

        // Upload records to device
        hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
        // Maintain a pointer to the device memory
        sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
        sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecordDirect);
        sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
    }


    void DirectLightPipeline::buildPipeline(OptixDeviceContext& context)
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
        OPTIX_CHECK(optixPipelineCreate(context,
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

    OptixTraversableHandle DirectLightPipeline::buildAccelerationStructure(OptixDeviceContext& context, GeometryBufferHandle& geometryBuffers, Scene& scene)
    {
        int bufferSize = scene.numObjects();

        OptixTraversableHandle asHandle{ 0 };

        // ==================================================================
        // Triangle inputs
        // ==================================================================
        std::vector<OptixBuildInput> triangleInput(bufferSize);
        std::vector<CUdeviceptr> d_vertices(bufferSize);
        std::vector<CUdeviceptr> d_indices(bufferSize);
        std::vector<uint32_t> triangleInputFlags(bufferSize);

        for (int meshID = 0; meshID < scene.numObjects(); meshID++) {
            // upload the model to the device: the builder
            std::shared_ptr<Model> model = scene.getGameObjects()[meshID]->model;
            std::shared_ptr<TriangleMesh> mesh = model->mesh;

            triangleInput[meshID] = {};
            triangleInput[meshID].type
                = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            // create local variables, because we need a *pointer* to the
            // device pointers
            d_vertices[meshID] = geometryBuffers.vertices[meshID].d_pointer();
            d_indices[meshID] = geometryBuffers.indices[meshID].d_pointer();

            triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
            triangleInput[meshID].triangleArray.numVertices = (int)model->mesh->vertices.size();
            triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

            triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(glm::ivec3);
            triangleInput[meshID].triangleArray.numIndexTriplets = (int)model->mesh->indices.size();
            triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];

            triangleInputFlags[meshID] = 0;

            // in this example we have one SBT entry, and no per-primitive
            // materials:
            triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
            triangleInput[meshID].triangleArray.numSbtRecords = 1;
            triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
            triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
            triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
        }

        // ==================================================================
        // BLAS setup
        // ==================================================================

        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
            | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accelOptions.motionOptions.numKeys = 1;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes blasBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage
        (context,
            &accelOptions,
            triangleInput.data(),
            bufferSize,  // num_build_inputs
            &blasBufferSizes
        ));

        // ==================================================================
        // prepare compaction
        // ==================================================================

        CUDABuffer compactedSizeBuffer;
        compactedSizeBuffer.alloc(sizeof(uint64_t));

        OptixAccelEmitDesc emitDesc;
        emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitDesc.result = compactedSizeBuffer.d_pointer();

        // ==================================================================
        // execute build (main stage)
        // ==================================================================

        CUDABuffer tempBuffer;
        tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

        CUDABuffer outputBuffer;
        outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

        OPTIX_CHECK(optixAccelBuild(context,
            /* stream */0,
            &accelOptions,
            triangleInput.data(),
            bufferSize,
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
        OPTIX_CHECK(optixAccelCompact(context,
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

}