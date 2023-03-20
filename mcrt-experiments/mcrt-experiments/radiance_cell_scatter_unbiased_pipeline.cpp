#include "radiance_cell_scatter_unbiased_pipeline.hpp"

namespace mcrt {
    extern "C" char embedded_ptx_code_radiance_cell_scattering_unbiased[];

    // SBT record for a raygen program
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecordRadianceCellScatter
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        void* data;
    };

    // SBT record for a miss program
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecordRadianceCellScatter
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        void* data;
    };

    // SBT record for a hitgroup program
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecordRadianceCellScatter
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        MeshSBTDataRadianceCellScatter data;
    };

    // TODO: We build GAS for each pipeline separately now, while most pipelines use the same, can't we reuse that one?
    RadianceCellScatterUnbiasedPipeline::RadianceCellScatterUnbiasedPipeline(OptixDeviceContext& context, GeometryBufferHandle& geometryBuffers, Scene& scene) : McrtPipeline(context, geometryBuffers, scene)
    {
        init(context, geometryBuffers, scene);

        std::vector<GeometryBufferHandle> geometries;
        geometries.push_back(geometryBuffers);
        int numSceneObjects = scene.numObjects();
        std::vector<int> numsBuildInputs = { numSceneObjects };
        std::vector<bool> disableAnyHit = { true };

        // Build GAS
        buildGASes(context, geometries, numsBuildInputs, disableAnyHit);
        launchParams.sceneTraversable = GASes[0].traversableHandle();

        launchParamsBuffer.alloc(sizeof(launchParams));
    }

    void RadianceCellScatterUnbiasedPipeline::uploadLaunchParams()
    {
        launchParamsBuffer.upload(&launchParams, 1);
    }

    void RadianceCellScatterUnbiasedPipeline::buildModule(OptixDeviceContext& context)
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
        pipelineLinkOptions.maxTraceDepth = 3;

        const std::string ptxCode = embedded_ptx_code_radiance_cell_scattering_unbiased;

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

    void RadianceCellScatterUnbiasedPipeline::buildDevicePrograms(OptixDeviceContext& context)
    {
        //---------------------------------------
        //  RAYGEN PROGRAMS
        //---------------------------------------
        raygenPGs.resize(1);

        OptixProgramGroupOptions pgOptionsRaygen = {};
        OptixProgramGroupDesc pgDescRaygen = {};
        pgDescRaygen.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pgDescRaygen.raygen.module = module;
        pgDescRaygen.raygen.entryFunctionName = "__raygen__renderFrame__cell__scattering__unbiased";

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
        pgDescMiss.miss.entryFunctionName = "__miss__radiance__cell__scattering__unbiased";

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

        pgDescHitgroupScene.hitgroup.entryFunctionNameCH = "__closesthit__radiance__cell__scattering__scene__unbiased";
        pgDescHitgroupScene.hitgroup.entryFunctionNameAH = "__anyhit__radiance__cell__scattering__scene__unbiased";

        OPTIX_CHECK(optixProgramGroupCreate(context,
            &pgDescHitgroupScene,
            1,
            &pgOptionsHitgroupScene,
            log, &sizeof_log,
            &hitgroupPGs[0]
        ));
    }

    void RadianceCellScatterUnbiasedPipeline::buildSBT(GeometryBufferHandle& geometryBuffers, Scene& scene)
    {
        // ----------------------------------------
      // Build raygen records
      // ----------------------------------------
        std::vector<RaygenRecordRadianceCellScatter> raygenRecords;
        for (int i = 0; i < raygenPGs.size(); i++) {
            RaygenRecordRadianceCellScatter rec;
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
        std::vector<MissRecordRadianceCellScatter> missRecords;
        for (int i = 0; i < missPGs.size(); i++) {
            MissRecordRadianceCellScatter rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
            rec.data = nullptr; /* for now ... */
            missRecords.push_back(rec);
        }
        // Upload records to device
        missRecordsBuffer.alloc_and_upload(missRecords);
        // Maintain a pointer to the device memory
        sbt.missRecordBase = missRecordsBuffer.d_pointer();

        sbt.missRecordStrideInBytes = sizeof(MissRecordRadianceCellScatter);
        sbt.missRecordCount = (int)missRecords.size();


        // ----------------------------------------
        // Build hitgroup records
        // ----------------------------------------
        int numObjects = scene.numObjects();
        std::vector<HitgroupRecordRadianceCellScatter> hitgroupRecords;
        for (int i = 0; i < numObjects; i++) {
            auto mesh = scene.getGameObjects()[i]->model->mesh;

            HitgroupRecordRadianceCellScatter rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0], &rec));

            rec.data.vertex = (glm::vec3*)geometryBuffers.vertices[i].d_pointer();
            rec.data.index = (glm::ivec3*)geometryBuffers.indices[i].d_pointer();
            rec.data.normal = (glm::vec3*)geometryBuffers.normals[i].d_pointer();
            rec.data.texcoord = (glm::vec2*)geometryBuffers.texCoords[i].d_pointer();
            rec.data.objectNr = i;

            hitgroupRecords.push_back(rec);
        }

        // Upload records to device
        hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
        // Maintain a pointer to the device memory
        sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
        sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecordRadianceCellScatter);
        sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
    }

    void RadianceCellScatterUnbiasedPipeline::buildPipeline(OptixDeviceContext& context)
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

    OptixTraversableHandle RadianceCellScatterUnbiasedPipeline::buildAccelerationStructure(OptixDeviceContext& context, GeometryBufferHandle& geometryBuffers, Scene& scene)
    {
        return NULL;
    }
}
