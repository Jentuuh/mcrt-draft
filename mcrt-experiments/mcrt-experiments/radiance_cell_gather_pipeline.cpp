#include "radiance_cell_gather_pipeline.hpp"
#include <iostream>

namespace mcrt {

    extern "C" char embedded_ptx_code_radiance_cell_gathering[];

    // SBT record for a raygen program
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecordRadianceCellGather
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        void* data;
    };

    // SBT record for a miss program
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecordRadianceCellGather
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        void* data;
    };

    // SBT record for a hitgroup program
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecordRadianceCellGather
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        MeshSBTDataRadianceCellGather data;
    };

	RadianceCellGatherPipeline::RadianceCellGatherPipeline(OptixDeviceContext& context, GeometryBufferHandle& radianceCellGeometry, GeometryBufferHandle& proxyGeometry, Scene& scene): McrtPipeline(context, radianceCellGeometry, scene)
	{
        initRadianceCellGather(context, radianceCellGeometry, proxyGeometry, scene);
        launchParams.traversable = buildAccelerationStructureRadianceCellGather(context, radianceCellGeometry, proxyGeometry, scene);
        launchParamsBuffer.alloc(sizeof(launchParams));
	}

    void RadianceCellGatherPipeline::uploadLaunchParams()
    {
        launchParamsBuffer.upload(&launchParams, 1);
    }

    void RadianceCellGatherPipeline::initRadianceCellGather(OptixDeviceContext& context, GeometryBufferHandle& radianceCellGeometry, GeometryBufferHandle& proxyGeometry, Scene& scene)
    {
        buildModule(context);
        buildDevicePrograms(context);
        buildPipeline(context);
        buildSBTRadianceCellGather(radianceCellGeometry, proxyGeometry, scene);
    }


    void RadianceCellGatherPipeline::buildModule(OptixDeviceContext& context)
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

        const std::string ptxCode = embedded_ptx_code_radiance_cell_gathering;

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

    void RadianceCellGatherPipeline::buildDevicePrograms(OptixDeviceContext& context)
    {
        //---------------------------------------
        //  RAYGEN PROGRAMS
        //---------------------------------------
        raygenPGs.resize(1);

        OptixProgramGroupOptions pgOptionsRaygen = {};
        OptixProgramGroupDesc pgDescRaygen = {};
        pgDescRaygen.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pgDescRaygen.raygen.module = module;
        pgDescRaygen.raygen.entryFunctionName = "__raygen__renderFrame__cell__gathering";

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
        pgDescMiss.miss.entryFunctionName = "__miss__radiance__cell__gathering";

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
        hitgroupPGs.resize(1);

        OptixProgramGroupOptions pgOptionsHitgroup = {};
        OptixProgramGroupDesc    pgDescHitgroup = {};
        pgDescHitgroup.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pgDescHitgroup.hitgroup.moduleCH = module;
        pgDescHitgroup.hitgroup.moduleAH = module;

        pgDescHitgroup.hitgroup.entryFunctionNameCH = "__closesthit__radiance__cell__gathering";
        pgDescHitgroup.hitgroup.entryFunctionNameAH = "__anyhit__radiance__cell__gathering";

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

    void RadianceCellGatherPipeline::buildSBT(GeometryBufferHandle& geometryBuffers, Scene& scene)
    {
        // No implementation in this subclass, this class has its own version of buildSBT
    }

    void RadianceCellGatherPipeline::buildSBTRadianceCellGather(GeometryBufferHandle& radianceCellGeometry, GeometryBufferHandle& proxyGeometry, Scene& scene)
    {
        // ----------------------------------------
        // Build raygen records
        // ----------------------------------------
        std::vector<RaygenRecordRadianceCellGather> raygenRecords;
        for (int i = 0; i < raygenPGs.size(); i++) {
            RaygenRecordRadianceCellGather rec;
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
        std::vector<MissRecordRadianceCellGather> missRecords;
        for (int i = 0; i < missPGs.size(); i++) {
            MissRecordRadianceCellGather rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
            rec.data = nullptr; /* for now ... */
            missRecords.push_back(rec);
        }
        // Upload records to device
        missRecordsBuffer.alloc_and_upload(missRecords);
        // Maintain a pointer to the device memory
        sbt.missRecordBase = missRecordsBuffer.d_pointer();
        sbt.missRecordStrideInBytes = sizeof(MissRecordRadianceCellGather);
        sbt.missRecordCount = (int)missRecords.size();


        // ----------------------------------------
        // Build hitgroup records
        // ----------------------------------------
        NonEmptyCells nonEmptyCells = scene.grid.getNonEmptyCells();
        int numNonEmptyCells = nonEmptyCells.nonEmptyCells.size();

        std::vector<HitgroupRecordRadianceCellGather> hitgroupRecords;

        // TODO: HIER MOET IK LOOPEN OVER DE CUBES DIE OBJECTEN BEVATTEN
        for (int i = 0; i < numNonEmptyCells; i++) {

            HitgroupRecordRadianceCellGather rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0], &rec));
            rec.data.cellIndex = i;
            rec.data.vertex = (glm::vec3*)radianceCellGeometry.vertices[nonEmptyCells.nonEmptyCellIndices[i]].d_pointer();     // Kan ik deze niet ook 1 keer opslaan en ter plekke hier transformeren?
            rec.data.index = (glm::ivec3*)radianceCellGeometry.indices[nonEmptyCells.nonEmptyCellIndices[i]].d_pointer();
            hitgroupRecords.push_back(rec);
        }

        // Upload records to device
        hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
        // Maintain a pointer to the device memory
        sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
        sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecordRadianceCellGather);
        sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
    }


    void RadianceCellGatherPipeline::buildPipeline(OptixDeviceContext& context)
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

    OptixTraversableHandle RadianceCellGatherPipeline::buildAccelerationStructure(OptixDeviceContext& context, GeometryBufferHandle& geometryBuffers, Scene& scene)
    {
        // No implementation in this subclass, this class has its own version of buildAccelerationStructure
        return NULL;
    }

    OptixTraversableHandle RadianceCellGatherPipeline::buildAccelerationStructureRadianceCellGather(OptixDeviceContext& context, GeometryBufferHandle& radianceCellGeometry, GeometryBufferHandle& proxyGeometry, Scene& scene)
    {
        // TODO: MAAK DIT HET AANTAL RADIANCE CELLS DIE LICHT BEVATTEN!
        NonEmptyCells nonEmptyCells = scene.grid.getNonEmptyCells();
        int numNonEmptyCells = nonEmptyCells.nonEmptyCells.size();

        OptixTraversableHandle asHandle{ 0 };

        // ==================================================================
        // Triangle inputs
        // ==================================================================
        std::vector<OptixBuildInput> triangleInput(numNonEmptyCells);
        std::vector<CUdeviceptr> d_vertices(numNonEmptyCells);
        std::vector<CUdeviceptr> d_indices(numNonEmptyCells);
        std::vector<uint32_t> triangleInputFlags(numNonEmptyCells);

        int verticesSize = (int)nonEmptyCells.nonEmptyCells[0]->getVertices().size();
        int indicesSize = (int)nonEmptyCells.nonEmptyCells[0]->getIndices().size();

        for (int meshID = 0; meshID < numNonEmptyCells; meshID++) {
            // upload the model to the device: the builder
            triangleInput[meshID] = {};
            triangleInput[meshID].type
                = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            // create local variables, because we need a *pointer* to the
            // device pointers
            d_vertices[meshID] = radianceCellGeometry.vertices[meshID].d_pointer();
            d_indices[meshID] = radianceCellGeometry.indices[meshID].d_pointer();

            triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
            triangleInput[meshID].triangleArray.numVertices = verticesSize;
            triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

            triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(glm::ivec3);
            triangleInput[meshID].triangleArray.numIndexTriplets = indicesSize;
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
            numNonEmptyCells,  // num_build_inputs
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
            numNonEmptyCells,
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