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

    RadianceCellGatherPipeline::RadianceCellGatherPipeline(OptixDeviceContext& context, GeometryBufferHandle& radianceCellGeometry, GeometryBufferHandle& proxyGeometry, Scene& scene) : McrtPipeline(context, radianceCellGeometry, scene)
    {
        initRadianceCellGather(context, radianceCellGeometry, proxyGeometry, scene);

        std::vector<GeometryBufferHandle> geometries;
        geometries.push_back(proxyGeometry);
        int numSceneObjects = scene.numObjects();
        std::vector<int> numsBuildInputs = { numSceneObjects };
        std::vector<bool> disableAnyHit = { true };

        // Build GAS
        buildGASes(context, geometries, numsBuildInputs, disableAnyHit);
        launchParams.sceneTraversable = GASes[0].traversableHandle();
        
        // Build IAS
        //std::vector<int>gasIndices = { 0,1 };
        //std::vector<glm::mat4> transforms = { glm::mat4(1.0f), glm::mat4(1.0f) };
        //buildIAS(context, transforms, GASes, 1, gasIndices);
        //launchParams.iasTraversable = ias->traversableHandle();

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
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        pipelineCompileOptions.usesMotionBlur = false;
        pipelineCompileOptions.numPayloadValues = 4;
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

        // Hitgroup scene geometry
        OptixProgramGroupOptions pgOptionsHitgroupScene = {};
        OptixProgramGroupDesc    pgDescHitgroupScene = {};
        pgDescHitgroupScene.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pgDescHitgroupScene.hitgroup.moduleCH = module;
        pgDescHitgroupScene.hitgroup.moduleAH = module;

        pgDescHitgroupScene.hitgroup.entryFunctionNameCH = "__closesthit__radiance__cell__gathering__scene";
        pgDescHitgroupScene.hitgroup.entryFunctionNameAH = "__anyhit__radiance__cell__gathering__scene";

        OPTIX_CHECK(optixProgramGroupCreate(context,
            &pgDescHitgroupScene,
            1,
            &pgOptionsHitgroupScene,
            log, &sizeof_log,
            &hitgroupPGs[0]
        ));

        // Hitgroup radiance grid geometry
        //OptixProgramGroupOptions pgOptionsHitgroupGrid = {};
        //OptixProgramGroupDesc    pgDescHitgroupGrid = {};
        //pgDescHitgroupGrid.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        //pgDescHitgroupGrid.hitgroup.moduleCH = module;
        //pgDescHitgroupGrid.hitgroup.moduleAH = module;

        //pgDescHitgroupGrid.hitgroup.entryFunctionNameCH = "__closesthit__radiance__cell__gathering__grid";
        //pgDescHitgroupGrid.hitgroup.entryFunctionNameAH = "__anyhit__radiance__cell__gathering__grid";

        //OPTIX_CHECK(optixProgramGroupCreate(context,
        //    &pgDescHitgroupGrid,
        //    1,
        //    &pgOptionsHitgroupGrid,
        //    log, &sizeof_log,
        //    &hitgroupPGs[1]
        //));

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
        std::vector<HitgroupRecordRadianceCellGather> hitgroupRecords;
        int numObjects = scene.numObjects();

        // Scene geometry
        for (int i = 0; i < numObjects; i++) {
            auto mesh = scene.getGameObjects()[i]->model->mesh;

            HitgroupRecordRadianceCellGather rec_scene;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0], &rec_scene));

            rec_scene.data.vertex = (glm::vec3*)proxyGeometry.vertices[i].d_pointer();
            rec_scene.data.index = (glm::ivec3*)proxyGeometry.indices[i].d_pointer();
            rec_scene.data.normal = (glm::vec3*)proxyGeometry.normals[i].d_pointer();
            rec_scene.data.texcoord = (glm::vec2*)proxyGeometry.texCoords[i].d_pointer();

            hitgroupRecords.push_back(rec_scene);
        }

        NonEmptyCells nonEmptyCells = scene.grid.getNonEmptyCells();
        int numNonEmptyCells = nonEmptyCells.nonEmptyCells.size();

        //// Radiance grid geometry
        //for (int i = 0; i < numNonEmptyCells; i++) {

        //    HitgroupRecordRadianceCellGather rec_grid;
        //    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[1], &rec_grid));

        //    rec_grid.data.cellIndex = i;
        //    rec_grid.data.vertex = (glm::vec3*)radianceCellGeometry.vertices[nonEmptyCells.nonEmptyCellIndices[i]].d_pointer();     // Kan ik deze niet ook 1 keer opslaan en ter plekke hier transformeren?
        //    rec_grid.data.index = (glm::ivec3*)radianceCellGeometry.indices[nonEmptyCells.nonEmptyCellIndices[i]].d_pointer();
        //    hitgroupRecords.push_back(rec_grid);
        //}

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
            2))
    }


    OptixTraversableHandle RadianceCellGatherPipeline::buildAccelerationStructure(OptixDeviceContext& context, GeometryBufferHandle& geometryBuffers, Scene& scene)
    {
        // No implementation in this subclass, this class has its own version of buildAccelerationStructure
        return NULL;
    }

}