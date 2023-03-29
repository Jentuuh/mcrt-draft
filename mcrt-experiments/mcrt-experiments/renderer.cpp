#include "renderer.hpp"

// This include may only appear in a SINGLE src file:
#include <optix_function_table_definition.h>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/string_cast.hpp>

// std
#include <iostream>
#include <cassert>
#include <fstream>
#include <random>

#include <stb/stb_image_write.h>


#define STRATIFIED_X_SIZE 5
#define STRATIFIED_Y_SIZE 5
#define SPHERICAL_HARMONIC_BASIS_FUNCTIONS 9
#define TEXTURE_DIVISION_RES 1024
#define IRRADIANCE_TEXTURE_RESOLUTION 2048


namespace mcrt {

    Renderer::Renderer(Scene& scene, Camera& camera, BIAS_MODE bias, PROBE_MODE probeType, IRRADIANCE_STORAGE_TYPE irradianceStorage): renderCamera{camera}, scene{scene}, irradStorageType{irradianceStorage}
    {
        initOptix();

        std::cout << "Creating OptiX context..." << std::endl;
        createContext();

        std::cout << "Filling geometry buffers..." << std::endl;
        fillGeometryBuffers();

        std::cout << "Loading textures..." << std::endl;
        createDiffuseTextures();

        Timer timer;

        std::cout << "Setting up pipelines..." << std::endl;
        GeometryBufferHandle geometryData = GeometryBufferHandle{ vertexBuffers, indexBuffers, normalBuffers, texcoordBuffers, diffuseTextureUVBuffers, diffuseTextureObjects, amountVertices, amountIndices };

        if (irradianceStorage == OCTREE_TEXTURE)
        {
            cameraPipelineOctree = std::make_unique<DefaultPipelineOctree>(optixContext, geometryData, scene);
            directLightPipelineOctree = std::make_unique<DirectLightPipelineOctree>(optixContext, geometryData, scene);

            if (bias == BIASED_PROBES)
            {
                if (probeType == CUBE_MAP)
                {
                    radianceCellGatherCubeMapPipelineOctree = std::make_unique<RadianceCellGatherCubeMapPipelineOctree>(optixContext, geometryData, scene);
                    radianceCellScatterCubeMapPipelineOctree = std::make_unique<RadianceCellScatterCubeMapPipelineOctree>(optixContext, geometryData, scene);
                }
                else if (probeType == SPHERICAL_HARMONICS)
                {
                  // ===========
                  // TODO
                  // ===========
                }
            }
            else if (bias == UNBIASED)
            {
                radianceCellScatterUnbiasedPipelineOctree = std::make_unique<RadianceCellScatterUnbiasedPipelineOctree>(optixContext, geometryData, scene);
            }

            // Initialize irradiance octree textures
            octreeTextures = std::make_unique<OctreeTexture>(10, scene);
            
            // TODO: find world positions per cell alternative (for octree)!
            //prepareUVsInsideBuffer();
            prepareWorldSamplePoints(octreeTextures->getLeafFaceArea());

            if (bias == BIASED_PROBES && probeType == CUBE_MAP)
            {
                initLightProbeCubeMaps(64, scene.grid.resolution.x);
            }

            calculateDirectLightingOctree();
            //calculateIndirectLightingOctree(bias, probeType);
        }
        else if (irradianceStorage == TEXTURE_2D) {
            timer.startTimedEvent("Pipeline creation");
            cameraPipeline = std::make_unique<DefaultPipeline>(optixContext, geometryData, scene);
            directLightPipeline = std::make_unique<DirectLightPipeline>(optixContext, geometryData, scene);

            if (bias == BIASED_PROBES)
            {
                if (probeType == CUBE_MAP)
                {
                    radianceCellGatherCubeMapPipeline = std::make_unique<RadianceCellGatherCubeMapPipeline>(optixContext, geometryData, scene);
                    radianceCellScatterCubeMapPipeline = std::make_unique<RadianceCellScatterCubeMapPipeline>(optixContext, geometryData, scene);
                }
                else if (probeType == SPHERICAL_HARMONICS)
                {
                    radianceCellGatherPipeline = std::make_unique<RadianceCellGatherPipeline>(optixContext, geometryData, scene); // For now we can use normal geometry instead of proxy
                    radianceCellScatterPipeline = std::make_unique<RadianceCellScatterPipeline>(optixContext, geometryData, scene);
                }
            }
            else if (bias == UNBIASED)
            {
                radianceCellScatterUnbiasedPipeline = std::make_unique<RadianceCellScatterUnbiasedPipeline>(optixContext, geometryData, scene);
            }
            timer.stopTimedEvent("Pipeline creation");

            timer.startTimedEvent("Irradiance texture creation");
            initLightingTexturesPerObject();
            timer.stopTimedEvent("Irradiance texture creation");
            //createWorldDataTextures();
            timer.startTimedEvent("UV world data generation/loading");
            prepareUVWorldPositionsPerObject(LOAD_WORLD_DATA);
            timer.stopTimedEvent("UV world data generation/loading");

            timer.startTimedEvent("Assigning UVs to cells");
            prepareUVsInsideBuffer();
            timer.stopTimedEvent("Assigning UVs to cells");


            if (bias == BIASED_PROBES && probeType == CUBE_MAP)
            {
                initLightProbeCubeMaps(64, scene.grid.resolution.x);
            }
            timer.startTimedEvent("Direct lighting");
            calculateDirectLighting();
            timer.stopTimedEvent("Direct lighting");

            timer.startTimedEvent("Indirect lighting");
            calculateIndirectLighting(bias, probeType);
            timer.stopTimedEvent("Indirect lighting");
        }

        timer.printSummary();
        std::cout << "Context, module, pipelines, etc, all set up." << std::endl;
        std::cout << "MCRT renderer fully set up." << std::endl;
    }

    void Renderer::fillGeometryBuffers()
    {
        // ======================
        //    NORMAL GEOMETRY
        // ======================
        int bufferSize = scene.numObjects();

        amountVertices.resize(bufferSize);
        amountIndices.resize(bufferSize);
        vertexBuffers.resize(bufferSize);
        indexBuffers.resize(bufferSize);
        normalBuffers.resize(bufferSize);
        texcoordBuffers.resize(bufferSize);
        diffuseTextureUVBuffers.resize(bufferSize);


        for (int meshID = 0; meshID < scene.numObjects(); meshID++) {
            // upload the model to the device: the builder
            std::shared_ptr<Model> model = scene.getGameObjects()[meshID]->model;
            std::shared_ptr<TriangleMesh> mesh = model->mesh;
            vertexBuffers[meshID].alloc_and_upload(scene.getGameObjects()[meshID]->getWorldVertices());
            amountVertices[meshID] = mesh->vertices.size();
            indexBuffers[meshID].alloc_and_upload(mesh->indices);
            amountIndices[meshID] = mesh->indices.size();
            if (!mesh->normals.empty())
                normalBuffers[meshID].alloc_and_upload(mesh->normals);
            if (!mesh->texCoords.empty())
                texcoordBuffers[meshID].alloc_and_upload(mesh->texCoords);
            if (!mesh->diffuseTextureCoords.empty())
                diffuseTextureUVBuffers[meshID].alloc_and_upload(mesh->diffuseTextureCoords);
        }

        // ============================
        //    RADIANCE CELL GEOMETRY
        // ============================
        NonEmptyCells nonEmpties = scene.grid.getNonEmptyCells();
        int numNonEmptyCells = nonEmpties.nonEmptyCells.size();

        radianceGridVertexBuffers.resize(numNonEmptyCells);
        radianceGridIndexBuffers.resize(numNonEmptyCells);
        amountVerticesRadianceGrid.resize(numNonEmptyCells);
        amountIndicesRadianceGrid.resize(numNonEmptyCells);

        for (int cellID = 0; cellID < numNonEmptyCells; cellID++) {
            radianceGridVertexBuffers[cellID].alloc_and_upload(nonEmpties.nonEmptyCells[cellID]->getVertices());
            radianceGridIndexBuffers[cellID].alloc_and_upload(nonEmpties.nonEmptyCells[cellID]->getIndices());
            amountVerticesRadianceGrid[cellID] = nonEmpties.nonEmptyCells[cellID]->getVertices().size();
            amountIndicesRadianceGrid[cellID] = nonEmpties.nonEmptyCells[cellID]->getIndices().size();
        }
    }

    void Renderer::createDiffuseTextures()
    {
        int numTextures = (int)scene.getDiffuseTextures().size();

        std::cout << "Found " << numTextures << " textures." << std::endl;

        textureArrays.resize(numTextures);
        diffuseTextureObjects.resize(numTextures);

        for (int textureID = 0; textureID < numTextures; textureID++) {
            auto texture = scene.getDiffuseTextures()[textureID];

            cudaResourceDesc res_desc = {};

            cudaChannelFormatDesc channel_desc;
            int32_t width = texture->resolution.x;
            int32_t height = texture->resolution.y;
            int32_t numComponents = 4;
            int32_t pitch = width * numComponents * sizeof(uint8_t);
            channel_desc = cudaCreateChannelDesc<uchar4>();

            cudaArray_t& pixelArray = textureArrays[textureID];
            CUDA_CHECK(MallocArray(&pixelArray,
                &channel_desc,
                width, height));

            CUDA_CHECK(Memcpy2DToArray(pixelArray,
                /* offset */0, 0,
                texture->pixel,
                pitch, pitch, height,
                cudaMemcpyHostToDevice));

            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = pixelArray;

            cudaTextureDesc tex_desc = {};
            tex_desc.addressMode[0] = cudaAddressModeWrap;
            tex_desc.addressMode[1] = cudaAddressModeWrap;
            tex_desc.filterMode = cudaFilterModeLinear;
            tex_desc.readMode = cudaReadModeNormalizedFloat;
            tex_desc.normalizedCoords = 1;
            tex_desc.maxAnisotropy = 1;
            tex_desc.maxMipmapLevelClamp = 99;
            tex_desc.minMipmapLevelClamp = 0;
            tex_desc.mipmapFilterMode = cudaFilterModePoint;
            tex_desc.borderColor[0] = 1.0f;
            tex_desc.sRGB = 0;

            // Create texture object
            cudaTextureObject_t cuda_tex = 0;
            CUDA_CHECK(CreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
            diffuseTextureObjects[textureID] = cuda_tex;
        }

        diffuseTextureObjectPointersBuffer.alloc_and_upload(diffuseTextureObjects);
    }

    void Renderer::createWorldDataTextures()
    {
        // ===================
        // World position data
        // ===================
        int numTextures = (int)scene.getWorldPosTextures().size();

        textureArrays.resize(numTextures);
        UVWorldPositionsTextures.resize(numTextures);

        for (int textureID = 0; textureID < numTextures; textureID++) {
            auto texture = scene.getWorldPosTextures()[textureID];

            for (int i = 8627066; i < 8627066 + 16; i++)
            {
                std::cout << texture->pixel[i] << std::endl;
            }

            cudaResourceDesc res_desc = {};

            cudaChannelFormatDesc channel_desc;
            int32_t width = texture->resolution.x;
            int32_t height = texture->resolution.y;
            int32_t pitch = width * sizeof(float4);;
            channel_desc = cudaCreateChannelDesc<float4>();

            //cudaArray_t& pixelArray = textureArrays[textureID];
            cudaArray* worldPosArray;

            CUDA_CHECK(MallocArray(&worldPosArray,
                &channel_desc,
                width, height));

            CUDA_CHECK(Memcpy2DToArray(worldPosArray,
                /* offset */0, 0,
                texture->pixel,
                pitch, pitch, height,
                cudaMemcpyHostToDevice));

            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = worldPosArray;

            cudaTextureDesc tex_desc = {};
            tex_desc.addressMode[0] = cudaAddressModeWrap;
            tex_desc.addressMode[1] = cudaAddressModeWrap;
            tex_desc.filterMode = cudaFilterModePoint;
            tex_desc.readMode = cudaReadModeElementType;
            tex_desc.normalizedCoords = 1;
            tex_desc.maxAnisotropy = 1;
            tex_desc.maxMipmapLevelClamp = 99;
            tex_desc.minMipmapLevelClamp = 0;
            tex_desc.mipmapFilterMode = cudaFilterModePoint;
            tex_desc.borderColor[0] = 1.0f;
            tex_desc.sRGB = 0;

            // Create texture object
            cudaTextureObject_t cuda_tex = 0;
            CUDA_CHECK(CreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
            UVWorldPositionsTextures[textureID] = cuda_tex;
        }

        // =================
        // World normal data
        // =================
        numTextures = (int)scene.getWorldNormalTextures().size();

        UVNormalsTextures.resize(numTextures);

        for (int textureID = 0; textureID < numTextures; textureID++) {
            auto texture = scene.getWorldNormalTextures()[textureID];

            cudaResourceDesc res_desc = {};

            cudaChannelFormatDesc channel_desc;
            int32_t width = texture->resolution.x;
            int32_t height = texture->resolution.y;
            int32_t pitch = width * sizeof(float4);;
            channel_desc = cudaCreateChannelDesc<float4>();

            cudaArray* worldNormalArray;

            CUDA_CHECK(MallocArray(&worldNormalArray,
                &channel_desc,
                width, height));

            CUDA_CHECK(Memcpy2DToArray(worldNormalArray,
                /* offset */0, 0,
                texture->pixel,
                pitch, pitch, height,
                cudaMemcpyHostToDevice));

            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = worldNormalArray;

            cudaTextureDesc tex_desc = {};
            tex_desc.addressMode[0] = cudaAddressModeWrap;
            tex_desc.addressMode[1] = cudaAddressModeWrap;
            tex_desc.filterMode = cudaFilterModePoint;
            tex_desc.readMode = cudaReadModeElementType;
            tex_desc.normalizedCoords = 1;
            tex_desc.maxAnisotropy = 1;
            tex_desc.maxMipmapLevelClamp = 99;
            tex_desc.minMipmapLevelClamp = 0;
            tex_desc.mipmapFilterMode = cudaFilterModePoint;
            tex_desc.borderColor[0] = 1.0f;
            tex_desc.sRGB = 0;

            // Create texture object
            cudaTextureObject_t cuda_tex = 0;
            CUDA_CHECK(CreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
            UVNormalsTextures[textureID] = cuda_tex;
        }

        // =============================
        // World diffuse tex coords data
        // =============================
        numTextures = (int)scene.getWorldDiffuseCoordsTextures().size();
        diffuseTextureUVsTextures.resize(numTextures);

        for (int textureID = 0; textureID < numTextures; textureID++) {
            auto texture = scene.getWorldNormalTextures()[textureID];

            cudaResourceDesc res_desc = {};

            cudaChannelFormatDesc channel_desc;
            int32_t width = texture->resolution.x;
            int32_t height = texture->resolution.y;
            int32_t pitch = width * sizeof(float4);;
            channel_desc = cudaCreateChannelDesc<float4>();

            cudaArray* diffuseTextureUVArray;

            CUDA_CHECK(MallocArray(&diffuseTextureUVArray,
                &channel_desc,
                width, height));

            CUDA_CHECK(Memcpy2DToArray(diffuseTextureUVArray,
                /* offset */0, 0,
                texture->pixel,
                pitch, pitch, height,
                cudaMemcpyHostToDevice));

            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = diffuseTextureUVArray;

            cudaTextureDesc tex_desc = {};
            tex_desc.addressMode[0] = cudaAddressModeWrap;
            tex_desc.addressMode[1] = cudaAddressModeWrap;
            tex_desc.filterMode = cudaFilterModePoint;
            tex_desc.readMode = cudaReadModeElementType;
            tex_desc.normalizedCoords = 1;
            tex_desc.maxAnisotropy = 1;
            tex_desc.maxMipmapLevelClamp = 99;
            tex_desc.minMipmapLevelClamp = 0;
            tex_desc.mipmapFilterMode = cudaFilterModePoint;
            tex_desc.borderColor[0] = 1.0f;
            tex_desc.sRGB = 0;

            // Create texture object
            cudaTextureObject_t cuda_tex = 0;
            CUDA_CHECK(CreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
            diffuseTextureUVsTextures[textureID] = cuda_tex;
        }
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


    // Render loop
    void Renderer::render()
    {
        if (irradStorageType == OCTREE_TEXTURE)
        {
            // First resize needs to be done before rendering
            if (cameraPipelineOctree->launchParams.frame.size.x == 0) return;

            // Irradiance octree textures
            cameraPipelineOctree->launchParams.octreeTextureDirect = (float*)octreeTextures->getOctreeGPUMemory().d_pointer();
            cameraPipelineOctree->launchParams.octreeTextureSecondBounce = (float*)octreeTextures->getOctreeGPUMemoryBounce2().d_pointer();

            cameraPipelineOctree->uploadLaunchParams();

            // Launch render pipeline
            OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                cameraPipelineOctree->pipeline, stream,
                /*! launch parameters and SBT */
                cameraPipelineOctree->launchParamsBuffer.d_pointer(),
                cameraPipelineOctree->launchParamsBuffer.sizeInBytes,
                &cameraPipelineOctree->sbt,
                /*! dimensions of the launch: */
                cameraPipelineOctree->launchParams.frame.size.x,
                cameraPipelineOctree->launchParams.frame.size.y,
                1
            ));


            // TODO: implement double buffering!!!
            // sync - make sure the frame is rendered before we download and
            // display (obviously, for a high-performance application you
            // want to use streams and double-buffering, but for this simple
            // example, this will have to do)
            CUDA_SYNC_CHECK();
        }
        else if (irradStorageType == TEXTURE_2D)
        {
            // First resize needs to be done before rendering
            if (cameraPipeline->launchParams.frame.size.x == 0) return;

            // Light bounce textures
            cameraPipeline->launchParams.directLightTextures = (cudaTextureObject_t*)directTextureObjectPointersBuffer.d_pointer();
            cameraPipeline->launchParams.secondBounceTextures = (cudaTextureObject_t*)secondBounceTextureObjectPointersBuffer.d_pointer();

            //cameraPipeline->launchParams.directLightTexture = textureObjectsDirect[0];
            //cameraPipeline->launchParams.secondBounceTexture = textureObjectsSecond[0];
            //cameraPipeline->launchParams.thirdBounceTexture = textureObjectsThird[0];

            cameraPipeline->uploadLaunchParams();

            // Launch render pipeline
            OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                cameraPipeline->pipeline, stream,
                /*! launch parameters and SBT */
                cameraPipeline->launchParamsBuffer.d_pointer(),
                cameraPipeline->launchParamsBuffer.sizeInBytes,
                &cameraPipeline->sbt,
                /*! dimensions of the launch: */
                cameraPipeline->launchParams.frame.size.x,
                cameraPipeline->launchParams.frame.size.y,
                1
            ));

            // TODO: implement double buffering!!!
            // sync - make sure the frame is rendered before we download and
            // display (obviously, for a high-performance application you
            // want to use streams and double-buffering, but for this simple
            // example, this will have to do)
            CUDA_SYNC_CHECK();
        }
    }



    void Renderer::updateCamera(GameObject* viewerObject)
    {
        if (irradStorageType == OCTREE_TEXTURE)
        {
            if (cameraPipelineOctree != nullptr)
            {
                //renderCamera = camera;
                cameraPipelineOctree->launchParams.camera.position = viewerObject->worldTransform.translation;
                cameraPipelineOctree->launchParams.camera.direction = normalize(renderCamera.getViewDirection(viewerObject->worldTransform.translation, viewerObject->worldTransform.rotation));
                const float cosFovy = 0.66f;
                const float aspect = float(cameraPipelineOctree->launchParams.frame.size.x) / float(cameraPipelineOctree->launchParams.frame.size.y);
                cameraPipelineOctree->launchParams.camera.horizontal
                    = cosFovy * aspect * normalize(renderCamera.getViewRight(viewerObject->worldTransform.translation, viewerObject->worldTransform.rotation));
                cameraPipelineOctree->launchParams.camera.vertical
                    = cosFovy * normalize(renderCamera.getViewUp(viewerObject->worldTransform.translation, viewerObject->worldTransform.rotation));
            }
        }
        else if (irradStorageType == TEXTURE_2D)
        {
            if (cameraPipeline != nullptr)
            {
                //renderCamera = camera;
                cameraPipeline->launchParams.camera.position = viewerObject->worldTransform.translation;
                cameraPipeline->launchParams.camera.direction = normalize(renderCamera.getViewDirection(viewerObject->worldTransform.translation, viewerObject->worldTransform.rotation));
                const float cosFovy = 0.66f;
                const float aspect = float(cameraPipeline->launchParams.frame.size.x) / float(cameraPipeline->launchParams.frame.size.y);
                cameraPipeline->launchParams.camera.horizontal
                    = cosFovy * aspect * normalize(renderCamera.getViewRight(viewerObject->worldTransform.translation, viewerObject->worldTransform.rotation));
                cameraPipeline->launchParams.camera.vertical
                    = cosFovy * normalize(renderCamera.getViewUp(viewerObject->worldTransform.translation, viewerObject->worldTransform.rotation));
            }
        }
    }


    void writeToImageUnsignedChar(std::string fileName, int resX, int resY, void* data)
    {
        stbi_write_png(fileName.c_str(), resX, resY, 4, data, resX * 4 * sizeof(stbi_uc));
    }


    void Renderer::initLightingTexturesPerObject()
    {   
        // Hardcoded for sponza scene
        std::vector<int> textureSizes{ 2048, 2048, 4096, 2048, 2048, 2048, 2048, 1024, 2048, 2048, 2048, 1024, 1024, 1024, 1024, 2048, 1024, 1024, 2048, 1024, 4096, 2048, 256, 256, 1024 };

        int numTextures = scene.getGameObjects().size();

        textureObjectsDirect.resize(numTextures);
        textureObjectsSecond.resize(numTextures);
        textureObjectsThird.resize(numTextures);

        surfaceObjectsDirect.resize(numTextures);
        surfaceObjectsSecond.resize(numTextures);
        surfaceObjectsThird.resize(numTextures);

        // Direct lighting
        for (int textureID = 0; textureID < numTextures; textureID++) {

            // Decide texture resolution based on the object's surface area
            //float objectArea = scene.getGameObjects()[textureID]->surfaceArea();
            //int size = objectArea * 512;
            //int size = 256;
            int size = textureSizes[textureID];
            size = GeneralUtils::pow2roundup(size);

            // We cannot have textures of 0x0
            if (size == 0)
                size = 1;

            std::cout << "Texture size: " << size << std::endl;

            std::vector<float> zeros(size * size * 4, 0.0f);

            cudaChannelFormatDesc channelDesc;
            channelDesc = cudaCreateChannelDesc<float4>();
            int32_t width = size;
            int32_t height = size;
            int32_t pitch = width * sizeof(float4);

            // Initialize CUDA array
            cudaArray* cuArray;
            CUDA_CHECK(MallocArray(&cuArray,
                &channelDesc,
                width,
                height,
                cudaArraySurfaceLoadStore));

            CUDA_CHECK(Memcpy2DToArray(cuArray,
                /* offset */0, 0,
                zeros.data(),
                pitch, pitch, height,
                cudaMemcpyHostToDevice));

            // Resource description for the CUDA array
            cudaResourceDesc resourceDesc = {};
            resourceDesc.resType = cudaResourceTypeArray;
            resourceDesc.res.array.array = cuArray;

            // Surface object creation
            cudaSurfaceObject_t cudaSurf = 0;
            CUDA_CHECK(CreateSurfaceObject(&cudaSurf, &resourceDesc));
            surfaceObjectsDirect[textureID] = cudaSurf;

            // Texture object creation
            cudaTextureDesc texDesc = {};
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 1;
            texDesc.maxAnisotropy = 1;
            texDesc.maxMipmapLevelClamp = 99;
            texDesc.minMipmapLevelClamp = 0;
            texDesc.mipmapFilterMode = cudaFilterModePoint;
            texDesc.borderColor[0] = 1.0f;
            texDesc.sRGB = 0;

            // Create texture object
            cudaTextureObject_t cudaTex = 0;
            CUDA_CHECK(CreateTextureObject(&cudaTex, &resourceDesc, &texDesc, nullptr));
            textureObjectsDirect[textureID] = cudaTex;

            // TEXTURE SIZES ONLY NEED TO BE PUSHED THE FIRST LOOP
            directTextureSizes.push_back(size);
        }

        // Second bounce
        for (int textureID = 0; textureID < numTextures; textureID++) {

            int size = directTextureSizes[textureID];

            std::vector<float> zeros(size * size * 4, 0.0f);

            cudaChannelFormatDesc channelDesc;
            channelDesc = cudaCreateChannelDesc<float4>();
            int32_t width = size;
            int32_t height = size;
            int32_t pitch = width * sizeof(float4);

            // Initialize CUDA array
            cudaArray* cuArray;
            CUDA_CHECK(MallocArray(&cuArray,
                &channelDesc,
                width,
                height,
                cudaArraySurfaceLoadStore));

            CUDA_CHECK(Memcpy2DToArray(cuArray,
                /* offset */0, 0,
                zeros.data(),
                pitch, pitch, height,
                cudaMemcpyHostToDevice));

            // Resource description for the CUDA array
            cudaResourceDesc resourceDesc = {};
            resourceDesc.resType = cudaResourceTypeArray;
            resourceDesc.res.array.array = cuArray;

            // Surface object creation
            cudaSurfaceObject_t cudaSurf = 0;
            CUDA_CHECK(CreateSurfaceObject(&cudaSurf, &resourceDesc));
            surfaceObjectsSecond[textureID] = cudaSurf;

            // Texture object creation
            cudaTextureDesc texDesc = {};
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 1;
            texDesc.maxAnisotropy = 1;
            texDesc.maxMipmapLevelClamp = 99;
            texDesc.minMipmapLevelClamp = 0;
            texDesc.mipmapFilterMode = cudaFilterModePoint;
            texDesc.borderColor[0] = 1.0f;
            texDesc.sRGB = 0;

            // Create texture object
            cudaTextureObject_t cudaTex = 0;
            CUDA_CHECK(CreateTextureObject(&cudaTex, &resourceDesc, &texDesc, nullptr));
            textureObjectsSecond[textureID] = cudaTex;
        }

        // Third bounce
        for (int textureID = 0; textureID < numTextures; textureID++) {

            int size = directTextureSizes[textureID];

            std::vector<float> zeros(size * size * 4, 0.0f);

            cudaChannelFormatDesc channelDesc;
            channelDesc = cudaCreateChannelDesc<float4>();
            int32_t width = size;
            int32_t height = size;
            int32_t pitch = width * sizeof(float4);

            // Initialize CUDA array
            cudaArray* cuArray;
            CUDA_CHECK(MallocArray(&cuArray,
                &channelDesc,
                width,
                height,
                cudaArraySurfaceLoadStore));

            CUDA_CHECK(Memcpy2DToArray(cuArray,
                /* offset */0, 0,
                zeros.data(),
                pitch, pitch, height,
                cudaMemcpyHostToDevice));

            // Resource description for the CUDA array
            cudaResourceDesc resourceDesc = {};
            resourceDesc.resType = cudaResourceTypeArray;
            resourceDesc.res.array.array = cuArray;

            // Surface object creation
            cudaSurfaceObject_t cudaSurf = 0;
            CUDA_CHECK(CreateSurfaceObject(&cudaSurf, &resourceDesc));
            surfaceObjectsThird[textureID] = cudaSurf;

            // Texture object creation
            cudaTextureDesc texDesc = {};
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 1;
            texDesc.maxAnisotropy = 1;
            texDesc.maxMipmapLevelClamp = 99;
            texDesc.minMipmapLevelClamp = 0;
            texDesc.mipmapFilterMode = cudaFilterModePoint;
            texDesc.borderColor[0] = 1.0f;
            texDesc.sRGB = 0;

            // Create texture object
            cudaTextureObject_t cudaTex = 0;
            CUDA_CHECK(CreateTextureObject(&cudaTex, &resourceDesc, &texDesc, nullptr));
            textureObjectsThird[textureID] = cudaTex;
        }

        // Buffers for pointers to all texture objects per lighting bounce
        directTextureObjectPointersBuffer.alloc_and_upload(textureObjectsDirect);
        secondBounceTextureObjectPointersBuffer.alloc_and_upload(textureObjectsSecond);
        thirdBounceTextureObjectPointersBuffer.alloc_and_upload(textureObjectsThird);

        directSurfaceObjectPointersBuffer.alloc_and_upload(surfaceObjectsDirect);
        secondSurfaceObjectPointersBuffer.alloc_and_upload(surfaceObjectsSecond);
        thirdSurfaceObjectPointersBuffer.alloc_and_upload(surfaceObjectsThird);

        // Buffer that stores the texture size of each game object (for now we assume we use the same resolution for each lighting bounce
        textureSizesBuffer.alloc_and_upload(directTextureSizes);
    }


    void Renderer::initLightProbeCubeMaps(int resolution, int gridRes)
    {
        // For a `gridRes x gridRes` cell grid, we have `gridRes x gridRes` light probes (in each cell center 1)
        std::vector<float> zeros(resolution * resolution * (gridRes) * (gridRes) * (gridRes) * 6 * 3, 0.0f);
        cubeMaps.alloc_and_upload(zeros);

        if (irradStorageType == OCTREE_TEXTURE)
        {
            radianceCellGatherCubeMapPipelineOctree->launchParams.cubeMaps = (float*)cubeMaps.d_pointer();
            radianceCellGatherCubeMapPipelineOctree->launchParams.cubeMapResolution = resolution;

            radianceCellScatterCubeMapPipelineOctree->launchParams.cubeMaps = (float*)cubeMaps.d_pointer();
            radianceCellScatterCubeMapPipelineOctree->launchParams.cubeMapResolution = resolution;
        }
        else if (irradStorageType == TEXTURE_2D)
        {
            radianceCellGatherCubeMapPipeline->launchParams.cubeMaps = (float*)cubeMaps.d_pointer();
            radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution = resolution;

            radianceCellScatterCubeMapPipeline->launchParams.cubeMaps = (float*)cubeMaps.d_pointer();
            radianceCellScatterCubeMapPipeline->launchParams.cubeMapResolution = resolution;
        }
    }

    void Renderer::calculateDirectLighting()
    {
        std::cout << "=======================================================" << std::endl;
        std::cout << "        CALCULATE DIRECT LIGHTING (TEXTURE 2D)         " << std::endl;
        std::cout << "=======================================================" << std::endl;

        // Get lights data from scene
        std::vector<LightData> lightData = scene.getLightsData();

        // Allocate device space for the light data buffer, then upload the light data to the device
        lightDataBuffer.resize(lightData.size() * sizeof(LightData));
        lightDataBuffer.upload(lightData.data(), 1);

        directLightPipeline->launchParams.amountLights = lightData.size();
        directLightPipeline->launchParams.lights = (LightData*)lightDataBuffer.d_pointer();
        directLightPipeline->launchParams.stratifyResX = STRATIFIED_X_SIZE;
        directLightPipeline->launchParams.stratifyResY = STRATIFIED_Y_SIZE;

        for (int o = 0; o < scene.getGameObjects().size(); o++)
        {
            // UV world data
            directLightPipeline->launchParams.uvPositions = UVWorldPositionsTextures[o];
            directLightPipeline->launchParams.uvNormals = UVNormalsTextures[o];
            directLightPipeline->launchParams.uvDiffuseColors = UVDiffuseColorTextures[o];
            directLightPipeline->launchParams.diffuseTextureUVs = diffuseTextureUVsTextures[o];
            
            // Diffuse color texture (per object)
            directLightPipeline->launchParams.hasTexture = scene.getGameObjects()[o]->model->mesh->diffuseTextureID > -1;
            if (directLightPipeline->launchParams.hasTexture) {
                directLightPipeline->launchParams.diffuseTexture = diffuseTextureObjects[o];
            }

            // Irradiance texture data
            directLightPipeline->launchParams.directLightingTexture = surfaceObjectsDirect[o];
            directLightPipeline->launchParams.textureSize = directTextureSizes[o];
            directLightPipeline->uploadLaunchParams();

            // Launch direct lighting pipeline
            OPTIX_CHECK(optixLaunch(
                directLightPipeline->pipeline, stream,
                directLightPipeline->launchParamsBuffer.d_pointer(),
                directLightPipeline->launchParamsBuffer.sizeInBytes,
                &directLightPipeline->sbt,
                directTextureSizes[o],                                              // dimension X: x resolution of UV map
                directTextureSizes[o],                                              // dimension Y: y resolution of UV map
                1                                                                   // dimension Z: 1
                // dimension X * dimension Y * dimension Z CUDA threads will be spawned 
            ));

            CUDA_SYNC_CHECK();            
        }

        //// Download resulting texture from GPU
        //std::vector<float>direct_lighting_result(directLightPipeline->launchParams.directLightingTexture.size * directLightPipeline->launchParams.directLightingTexture.size * 3);
        //directLightingTexture.download(direct_lighting_result.data(),
        //        directLightPipeline->launchParams.directLightingTexture.size * directLightPipeline->launchParams.directLightingTexture.size * 3);

        //HDRImage hdr{ directLightPipeline->launchParams.directLightingTexture.size ,directLightPipeline->launchParams.directLightingTexture.size, direct_lighting_result.data() };
        //hdr.saveImage("../debug_output/testHDR.hdr");
        //
        //std::vector<uint32_t> direct_lighting_result(directLightPipeline->launchParams.directLightingTexture.size * directLightPipeline->launchParams.directLightingTexture.size);
        //directLightingTexture.download(direct_lighting_result.data(),
        //    directLightPipeline->launchParams.directLightingTexture.size * directLightPipeline->launchParams.directLightingTexture.size);


        //// Flip Y coordinates, otherwise UV map is upside down
        //std::vector<uint32_t> result_reversed;

        //for (int y = directLightPipeline->launchParams.directLightingTexture.size - 1; y >= 0; y--)
        //{
        //    for (int x = 0; x < directLightPipeline->launchParams.directLightingTexture.size; x++)
        //    {
        //        result_reversed.push_back(direct_lighting_result[y * directLightPipeline->launchParams.directLightingTexture.size + x]);
        //    }
        //}


        //std::reverse(direct_lighting_result.begin(), direct_lighting_result.end());
        // Write the result to an image (for debugging purposes)
        // writeToImage("direct_lighting_output.png", directLightPipeline->launchParams.directLightingTexture.size, directLightPipeline->launchParams.directLightingTexture.size, result_reversed.data());
    }


    void Renderer::calculateIndirectLighting(BIAS_MODE bias, PROBE_MODE mode)
    {
        if (bias == BIASED_PROBES)
        {
            if (mode == SPHERICAL_HARMONICS)
            {
                std::cout << "=======================================================" << std::endl;
                std::cout << "        INDIRECT LIGHTING CALCULATION (SH)             " << std::endl;
                std::cout << "=======================================================" << std::endl;

                for (int i = 0; i < 1; i++)
                {
                    std::cout << "Calculating radiance cell gather pass " << i << "..." << std::endl;

                    switch (i)
                    {
                    case 0:
                        //calculateRadianceCellGatherPass(directLightingTexture);
                        std::cout << "Calculating radiance cell scatter pass " << i << "..." << std::endl;
                        //calculateRadianceCellScatterPass(i, secondBounceTexture);
                        break;
                    case 1:
                        //calculateRadianceCellGatherPass(secondBounceTexture);
                        std::cout << "Calculating radiance cell scatter pass " << i << "..." << std::endl;
                        //calculateRadianceCellScatterPass(i, thirdBounceTexture);
                        break;
                    default:
                        break;
                    }
                }
            }
            else if (mode == CUBE_MAP)
            {
                std::cout << "====================================================" << std::endl;
                std::cout << "      INDIRECT LIGHTING CALCULATION (CUBEMAP)       " << std::endl;
                std::cout << "====================================================" << std::endl;

                for (int i = 0; i < 1; i++)
                {
                    std::cout << "Calculating radiance cell gather pass " << i << "..." << std::endl;

                    switch (i)
                    {
                    case 0:
                        calculateRadianceCellGatherPassCubeMapAlt((cudaTextureObject_t*)directTextureObjectPointersBuffer.d_pointer());
                        std::cout << "Calculating radiance cell scatter pass " << i << "..." << std::endl;
                        calculateRadianceCellScatterPassCubeMap(i, (cudaTextureObject_t*)directTextureObjectPointersBuffer.d_pointer(), (cudaSurfaceObject_t*)secondSurfaceObjectPointersBuffer.d_pointer());
                        //lightProbeTest(i, directLightingTexture, secondBounceTexture);
                        //octreeTextureTest();
                        //textureAndSurfaceObjectTest();

                        break;
                    case 1:
                        calculateRadianceCellGatherPassCubeMapAlt((cudaTextureObject_t*)secondBounceTextureObjectPointersBuffer.d_pointer());
                        std::cout << "Calculating radiance cell scatter pass " << i << "..." << std::endl;
                        calculateRadianceCellScatterPassCubeMap(i, (cudaTextureObject_t*)secondBounceTextureObjectPointersBuffer.d_pointer(), (cudaSurfaceObject_t*)thirdSurfaceObjectPointersBuffer.d_pointer());
                        break;
                    default:
                        break;
                    }
                }
            }
        }
        else if (bias == UNBIASED)
        {
            std::cout << "==================================================" << std::endl;
            std::cout << "     INDIRECT LIGHTING CALCULATION (UNBIASED)     " << std::endl;
            std::cout << "==================================================" << std::endl;

            for (int i = 0; i < 2; i++)
            {   
                std::cout << "Unbiased approach iteration " << i << "..." << std::endl;
                switch (i)
                {
                case 0:
                    calculateRadianceCellScatterUnbiased(i, (cudaTextureObject_t*)directTextureObjectPointersBuffer.d_pointer(), (cudaSurfaceObject_t*)secondSurfaceObjectPointersBuffer.d_pointer());
                    break;
                case 1:
                    calculateRadianceCellScatterUnbiased(i, (cudaTextureObject_t*)secondBounceTextureObjectPointersBuffer.d_pointer(), (cudaSurfaceObject_t*)thirdSurfaceObjectPointersBuffer.d_pointer());
                    break;
                default:
                    break;
                }
            }
        }
    }

    void Renderer::initSHWeightsBuffer(int amountNonEmptyCells)
    {        
        SHWeightsDataBuffer.free();
        // How to index: (nonEmptyCellIndex * 8 * SPHERICAL_HARMONIC_BASIS_FUNCTIONS) + (SHIndex * SPHERICAL_HARMONIC_BASIS_FUNCTIONS) + BasisFunctionIndex
        std::vector<float> shCoefficients(amountNonEmptyCells * 8 * SPHERICAL_HARMONIC_BASIS_FUNCTIONS, 0.0);
        SHWeightsDataBuffer.alloc_and_upload(shCoefficients);
        radianceCellGatherPipeline->launchParams.sphericalHarmonicsWeights.weights = (float*)SHWeightsDataBuffer.d_pointer();
        radianceCellGatherPipeline->launchParams.sphericalHarmonicsWeights.size = amountNonEmptyCells * 8 * SPHERICAL_HARMONIC_BASIS_FUNCTIONS; // 8 SHs per cell, each 9 basis functions
        radianceCellGatherPipeline->launchParams.sphericalHarmonicsWeights.amountBasisFunctions = SPHERICAL_HARMONIC_BASIS_FUNCTIONS;

        numSamplesAccumulatorsBuffer.free();
        std::vector<int> numSamplesAccumulator(amountNonEmptyCells * 8);
        numSamplesAccumulatorsBuffer.alloc_and_upload(numSamplesAccumulator);
        radianceCellGatherPipeline->launchParams.shNumSamplesAccumulators = (int*)numSamplesAccumulatorsBuffer.d_pointer();

        std::cout << "Size of SH weights buffer on GPU in bytes: " << shCoefficients.size() * sizeof(float) << std::endl;
    }

    void Renderer::initSHAccumulators(int divisionResolution, int amountNonEmptyCells)
    {
        // Initialize SH weights accumulators
        std::vector<float> shAccumulators(amountNonEmptyCells * divisionResolution * divisionResolution * 8 * SPHERICAL_HARMONIC_BASIS_FUNCTIONS, 0.0);
        SHAccumulatorsBuffer.alloc_and_upload(shAccumulators);
        radianceCellGatherPipeline->launchParams.shAccumulators = (float*)SHAccumulatorsBuffer.d_pointer();

        // Initialize SH num samples accumulators
        std::vector<int> numSamplesAccumulators(amountNonEmptyCells * 8 * divisionResolution * divisionResolution, 0);
        numSamplesAccumulatorsBuffer.alloc_and_upload(numSamplesAccumulators);
        radianceCellGatherPipeline->launchParams.shNumSamplesAccumulators = (int*)numSamplesAccumulatorsBuffer.d_pointer();
    }


    void Renderer::calculateRadianceCellGatherPass(CUDABuffer& previousPassLightSourceTexture)
    {
        // TODO: For now we're using the same texture size as for the direct lighting pass, we can downsample in the future to gain performance
        const int texSize = directTextureSizes[0];

        // Initialize non-empty cell data on GPU
        NonEmptyCells nonEmpties = scene.grid.getNonEmptyCells();
        std::vector<glm::vec3> nonEmptyCellCenters;
        for (auto& nonEmpty : nonEmpties.nonEmptyCells)
        {
            nonEmptyCellCenters.push_back(nonEmpty->getCenter());
        }
        nonEmptyCellDataBuffer.resize(nonEmptyCellCenters.size() * sizeof(glm::vec3));
        nonEmptyCellDataBuffer.upload(nonEmptyCellCenters.data(), nonEmptyCellCenters.size());
        radianceCellGatherPipeline->launchParams.nonEmptyCells.centers = (glm::vec3*)nonEmptyCellDataBuffer.d_pointer();
        radianceCellGatherPipeline->launchParams.nonEmptyCells.size = nonEmpties.nonEmptyCells.size();

        std::cout << "Amount non-empty cells: " << nonEmpties.nonEmptyCells.size() << std::endl;

        // Initialize Light Source Texture data on GPU
        // Take the direct lighting pass output as light source texture (for now), this should normally be the output from the previous pass in each iteration
        radianceCellGatherPipeline->launchParams.lightSourceTexture.colorBuffer = (float*)previousPassLightSourceTexture.d_pointer();
        radianceCellGatherPipeline->launchParams.lightSourceTexture.size = texSize;

        // Initialize SH data on GPU
        initSHWeightsBuffer(nonEmptyCellCenters.size());

        //initSHAccumulators(TEXTURE_DIVISION_RES, nonEmptyCellCenters.size());

        // Initialize UV World positions data on GPU
        radianceCellGatherPipeline->launchParams.uvWorldPositions.size = texSize * texSize;
        radianceCellGatherPipeline->launchParams.uvWorldPositions.UVDataBuffer = (UVWorldData*)samplePointWorldPositionDeviceBuffer.d_pointer();
        
        // Initialize cell size in launch params
        radianceCellGatherPipeline->launchParams.cellSize = scene.grid.getCellSize();

        // Initialize stratify cell sizes in launch params
        radianceCellGatherPipeline->launchParams.stratifyResX = STRATIFIED_X_SIZE;
        radianceCellGatherPipeline->launchParams.stratifyResY = STRATIFIED_Y_SIZE;

        radianceCellGatherPipeline->launchParams.divisionResolution = TEXTURE_DIVISION_RES;

        for (int i = 0; i < nonEmpties.nonEmptyCells.size(); i++)
        {
            radianceCellGatherPipeline->launchParams.nonEmptyCellIndex = i;
            radianceCellGatherPipeline->uploadLaunchParams();

            OPTIX_CHECK(optixLaunch(
                radianceCellGatherPipeline->pipeline, stream,
                radianceCellGatherPipeline->launchParamsBuffer.d_pointer(),
                radianceCellGatherPipeline->launchParamsBuffer.sizeInBytes,
                &radianceCellGatherPipeline->sbt,
                TEXTURE_DIVISION_RES,                                               // dimension X: divisionResX: amount of tiles in the X direction of the light src texture
                TEXTURE_DIVISION_RES,                                               // dimension Y: divisionResY: amount of tiles in the Y direction of the light src texture
                1
                // dimension X * dimension Y * dimension Z CUDA threads will be spawned 
            ));
        }


        CUDA_SYNC_CHECK();

        // Print SH results
        std::vector<float> shCoefficients(nonEmptyCellCenters.size() * 8 * SPHERICAL_HARMONIC_BASIS_FUNCTIONS);
        SHWeightsDataBuffer.download(shCoefficients.data(), nonEmptyCellCenters.size() * 8 * SPHERICAL_HARMONIC_BASIS_FUNCTIONS);

        std::vector<int> numSamplesPerSH(nonEmptyCellCenters.size() * 8);
        numSamplesAccumulatorsBuffer.download(numSamplesPerSH.data(), nonEmptyCellCenters.size() * 8);

        // Write SH weights to file
        writeWeightsToTxtFile(shCoefficients, numSamplesPerSH, nonEmptyCellCenters.size());

        for (int i = 0; i < nonEmptyCellCenters.size(); i++)
        {
            int cellOffset = i * 8 * SPHERICAL_HARMONIC_BASIS_FUNCTIONS;
            for (int sh = 0; sh < 8; sh++)
            {
                if (numSamplesPerSH[i * 8 + sh] > 0)
                {
                    float weight = 1.0f / (float(numSamplesPerSH[i * 8 + sh]) * 4.0f * glm::pi<float>()); // TODO: Not sure if the inverse weight should be 4 pi (probably not, because we're not uniformly sampling on a sphere, but rather only the light sources in the scene)
                    int shOffset = sh * SPHERICAL_HARMONIC_BASIS_FUNCTIONS;
                    std::cout << "[";
                    for (int bf = 0; bf < 9; bf++)
                    {
                        std::cout << shCoefficients[cellOffset + shOffset + bf] * weight << ",";
                        shCoefficients[cellOffset + shOffset + bf] *= weight;
                    }
                    std::cout << "]" << std::endl;
                }  
            }
        }
        // Upload back to GPU
        SHWeightsDataBuffer.upload(shCoefficients.data(), shCoefficients.size());
    }   

    // Cubemap alternative approach avoids 'holes' in cubemap texture by iterating over the cubemap pixels instead of the surrounding indirect light source texels.
    void Renderer::calculateRadianceCellGatherPassCubeMapAlt(cudaTextureObject_t* previousPassLightSourceTextures)
    {
        const float cellSize = scene.grid.getCellSize();

        // Initialize cell size in launch params
        radianceCellGatherCubeMapPipeline->launchParams.cellSize = cellSize;

        // Initialize Light Source Textures data on GPU
        radianceCellGatherCubeMapPipeline->launchParams.lightSourceTextures = previousPassLightSourceTextures;

        for (int z = 0; z < scene.grid.resolution.z; z++)
        {
            for (int y = 0; y < scene.grid.resolution.y; y++)
            {
                for (int x = 0; x < scene.grid.resolution.x; x++)
                {
                    int currentProbeOffset = ((z * scene.grid.resolution.x * scene.grid.resolution.y) + (y * scene.grid.resolution.x) + (x)) * 6 * radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution * radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution;
                    radianceCellGatherCubeMapPipeline->launchParams.probeOffset = currentProbeOffset;
                    radianceCellGatherCubeMapPipeline->launchParams.probePosition = glm::vec3{ x * cellSize + (0.5f * cellSize), y * cellSize + (0.5f * cellSize), z * cellSize + (0.5f * cellSize) }; // Probes are centered in each radiance cell

                    radianceCellGatherCubeMapPipeline->uploadLaunchParams();

                    OPTIX_CHECK(optixLaunch(
                        radianceCellGatherCubeMapPipeline->pipeline, stream,
                        radianceCellGatherCubeMapPipeline->launchParamsBuffer.d_pointer(),
                        radianceCellGatherCubeMapPipeline->launchParamsBuffer.sizeInBytes,
                        &radianceCellGatherCubeMapPipeline->sbt,
                        radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution,          // dimension X: divisionResX: width of cubemap face texture
                        radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution,          // dimension Y: divisionResY: height of cubemap face texture
                        6                                                                           // dimension Z: amount of cubemap faces
                        // dimension X * dimension Y * dimension Z CUDA threads will be spawned 
                    ));

                    CUDA_SYNC_CHECK();
                                 
                }
            }
        }

        //// Visualize cubemap from a probe as a test
        //glm::ivec3 testProbeCoord = scene.grid.getCell(12).getCellCoords();
        //int testProbeOffset = ((testProbeCoord.z * scene.grid.resolution.x * scene.grid.resolution.y) + (testProbeCoord.y * scene.grid.resolution.x) + (testProbeCoord.x)) * 6 * radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution * radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution;
        //for (int f = 0; f < 6; f++)
        //{
        //    int faceOffset = f * radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution * radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution;
        //    int totalOffset = testProbeOffset + faceOffset;
        //    std::vector<uint32_t> cubeMapFace(radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution * radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution);
        //    cubeMaps.download_with_offset(cubeMapFace.data(), radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution * radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution, totalOffset);
        //    writeToImage("cubemap_face_" + std::to_string(f) + ".png", radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution, radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution, cubeMapFace.data());
        //}

        //// Visualize cubemap 2 from a probe as a test
        //glm::ivec3 testProbeCoord2 = scene.grid.getCell(12).getCellCoords() + glm::ivec3{ 2, 0, 0 };
        //int testProbeOffset2 = ((testProbeCoord2.z * scene.grid.resolution.x * scene.grid.resolution.y) + (testProbeCoord2.y * scene.grid.resolution.x) + (testProbeCoord2.x)) * 6 * radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution * radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution;
        //for (int f = 0; f < 6; f++)
        //{
        //    int faceOffset = f * radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution * radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution;
        //    int totalOffset = testProbeOffset2 + faceOffset;
        //    std::vector<uint32_t> cubeMapFace(radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution * radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution);
        //    cubeMaps.download_with_offset(cubeMapFace.data(), radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution * radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution, totalOffset);
        //    writeToImage("cubemap2_face_" + std::to_string(f) + ".png", radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution, radianceCellGatherCubeMapPipeline->launchParams.cubeMapResolution, cubeMapFace.data());
        //}
    }


    void Renderer::calculateRadianceCellScatterPass(int iteration, CUDABuffer& dstTexture)
    {
        // TODO: For now we're using the same texture size as for the direct lighting pass, we can downsample in the future to gain performance
        const int texSize = directTextureSizes[0];

        NonEmptyCells nonEmpties = scene.grid.getNonEmptyCells();
       
        for (int i = 0 ; i < nonEmpties.nonEmptyCells.size(); i++)
        {
            radianceCellScatterPipeline->launchParams.nonEmptyCellIndex = i;

            radianceCellScatterPipeline->launchParams.currentBounceTexture.size = texSize;
            radianceCellScatterPipeline->launchParams.currentBounceTexture.colorBuffer = (float*)dstTexture.d_pointer();

            // Load uvs per cell 
            std::vector<glm::vec2> cellUVs = nonEmpties.nonEmptyCells[i]->getUVsInside();

            std::cout << "UVs in this cell: " << cellUVs.size() << std::endl;

            // SH weights data
            radianceCellScatterPipeline->launchParams.sphericalHarmonicsWeights.weights = (float*)SHWeightsDataBuffer.d_pointer();
            radianceCellScatterPipeline->launchParams.sphericalHarmonicsWeights.size = nonEmpties.nonEmptyCells.size() * 8 * SPHERICAL_HARMONIC_BASIS_FUNCTIONS; // 8 SHs per cell, each 9 basis functions
            radianceCellScatterPipeline->launchParams.sphericalHarmonicsWeights.amountBasisFunctions = SPHERICAL_HARMONIC_BASIS_FUNCTIONS;

            // Radiance cell data
            radianceCellScatterPipeline->launchParams.cellCenter = nonEmpties.nonEmptyCells[i]->getCenter();
            radianceCellScatterPipeline->launchParams.cellSize = scene.grid.getCellSize();

            // UV world position data
            radianceCellScatterPipeline->launchParams.uvWorldPositions.size = texSize;
            radianceCellScatterPipeline->launchParams.uvWorldPositions.UVDataBuffer = (UVWorldData*)samplePointWorldPositionDeviceBuffer.d_pointer();

            // Stratified sampling parameters
            radianceCellScatterPipeline->launchParams.stratifyResX = STRATIFIED_X_SIZE;
            radianceCellScatterPipeline->launchParams.stratifyResY = STRATIFIED_Y_SIZE;

            radianceCellScatterPipeline->uploadLaunchParams();

            OPTIX_CHECK(optixLaunch(
                radianceCellScatterPipeline->pipeline, stream,
                radianceCellScatterPipeline->launchParamsBuffer.d_pointer(),
                radianceCellScatterPipeline->launchParamsBuffer.sizeInBytes,
                &radianceCellScatterPipeline->sbt,
                cellUVs.size(),                     // dimension X: amount of UV texels in the cell
                1,                                  // dimension Y: 1
                1                                   // dimension Z: 1
                // dimension X * dimension Y * dimension Z CUDA threads will be spawned 
            ));

            CUDA_SYNC_CHECK();
        }

        // Download resulting texture from GPU
        std::vector<uint32_t> current_bounce_result(radianceCellScatterPipeline->launchParams.currentBounceTexture.size * radianceCellScatterPipeline->launchParams.currentBounceTexture.size);
        dstTexture.download(current_bounce_result.data(),
            radianceCellScatterPipeline->launchParams.currentBounceTexture.size * radianceCellScatterPipeline->launchParams.currentBounceTexture.size);

        // Write the result to an image (for debugging purposes)
        //writeToImage("current_bounce_output" + std::to_string(iteration) + ".png", radianceCellScatterPipeline->launchParams.currentBounceTexture.size, radianceCellScatterPipeline->launchParams.currentBounceTexture.size, current_bounce_result.data());
    }

    void Renderer::calculateRadianceCellScatterPassCubeMap(int iteration, cudaTextureObject_t* prevBounceTexture, cudaSurfaceObject_t* dstTexture)
    {

        // TODO: For now we're using the same texture size as for the direct lighting pass, we can downsample in the future to gain performance
        const int texSize = directTextureSizes[0];
        NonEmptyCells nonEmpties = scene.grid.getNonEmptyCells();

        for (int i = 0; i < nonEmpties.nonEmptyCells.size(); i++)
        {
            radianceCellScatterCubeMapPipeline->launchParams.nonEmptyCellIndex = i;
            radianceCellScatterCubeMapPipeline->launchParams.cellCoords = nonEmpties.nonEmptyCells[i]->getCellCoords();
            radianceCellScatterCubeMapPipeline->launchParams.probeWidthRes = scene.grid.resolution.x;
            radianceCellScatterCubeMapPipeline->launchParams.probeHeightRes = scene.grid.resolution.y;

            // Texture that we write to
            radianceCellScatterCubeMapPipeline->launchParams.currentBounceTextures = dstTexture;

            // Indirect light source texture that we read from (in case of local ray tracing)
            radianceCellScatterCubeMapPipeline->launchParams.prevBounceTextures = prevBounceTexture;

            // Load uvs per cell 
            std::vector<glm::vec2> cellUVs = nonEmpties.nonEmptyCells[i]->getUVsInside();
            std::cout << "Iteration " << i << ": UVs in this cell: " << cellUVs.size() << std::endl;

            // Radiance cell data
            radianceCellScatterCubeMapPipeline->launchParams.cellCenter = nonEmpties.nonEmptyCells[i]->getCenter();
            radianceCellScatterCubeMapPipeline->launchParams.cellSize = scene.grid.getCellSize();

            // UV world position data
            radianceCellScatterCubeMapPipeline->launchParams.uvPositions = (cudaTextureObject_t*) uvWorldPositionTextureObjectPointersBuffer.d_pointer();
            radianceCellScatterCubeMapPipeline->launchParams.uvNormals = (cudaTextureObject_t*) uvNormalTextureObjectPointersBuffer.d_pointer();
            radianceCellScatterCubeMapPipeline->launchParams.uvDiffuseColors = (cudaTextureObject_t*) uvDiffuseColorTextureObjectPointersBuffer.d_pointer();

            radianceCellScatterCubeMapPipeline->launchParams.objectTextureResolutions = (int*) textureSizesBuffer.d_pointer();

            // Diffuse texture data
            radianceCellScatterCubeMapPipeline->launchParams.hasTexture = (int*) objectHasTextureBuffer.d_pointer();
            radianceCellScatterCubeMapPipeline->launchParams.diffuseTextures = (cudaTextureObject_t*) diffuseTextureObjectPointersBuffer.d_pointer();
            radianceCellScatterCubeMapPipeline->launchParams.diffuseTextureUVs = (cudaTextureObject_t*)diffuseTextureUVsTextureObjectPointersBuffer.d_pointer();

            radianceCellScatterCubeMapPipeline->uploadLaunchParams();

            OPTIX_CHECK(optixLaunch(
                radianceCellScatterCubeMapPipeline->pipeline, stream,
                radianceCellScatterCubeMapPipeline->launchParamsBuffer.d_pointer(),
                radianceCellScatterCubeMapPipeline->launchParamsBuffer.sizeInBytes,
                &radianceCellScatterCubeMapPipeline->sbt,
                cellUVs.size(),                     // dimension X: amount of UV texels in the cell
                1,                                  // dimension Y: 1
                1                                   // dimension Z: 1
                // dimension X * dimension Y * dimension Z CUDA threads will be spawned 
            ));

            CUDA_SYNC_CHECK();
        }

        //// Download resulting texture from GPU
        //std::vector<uint32_t> current_bounce_result(radianceCellScatterCubeMapPipeline->launchParams.currentBounceTexture.size * radianceCellScatterCubeMapPipeline->launchParams.currentBounceTexture.size);
        //dstTexture.download(current_bounce_result.data(),
        //    radianceCellScatterCubeMapPipeline->launchParams.currentBounceTexture.size * radianceCellScatterCubeMapPipeline->launchParams.currentBounceTexture.size);

        //// Write the result to an image (for debugging purposes)
        //writeToImage("current_bounce_output" + std::to_string(iteration) + ".png", radianceCellScatterCubeMapPipeline->launchParams.currentBounceTexture.size, radianceCellScatterCubeMapPipeline->launchParams.currentBounceTexture.size, current_bounce_result.data());
    }

    void Renderer::lightProbeTest(int iteration, CUDABuffer& prevBounceTexture, CUDABuffer& dstTexture)
    {
        std::cout << "LIGHT PROBE TESTING... " << std::endl;

        // TODO: For now we're using the same texture size as for the direct lighting pass, we can downsample in the future to gain performance
        const int texSize = directTextureSizes[0];
        NonEmptyCells nonEmpties = scene.grid.getNonEmptyCells();

        radianceCellScatterCubeMapPipeline->launchParams.nonEmptyCellIndex = 12;
        radianceCellScatterCubeMapPipeline->launchParams.cellCoords = scene.grid.getCell(12).getCellCoords();
        radianceCellScatterCubeMapPipeline->launchParams.probeWidthRes = scene.grid.resolution.x;
        radianceCellScatterCubeMapPipeline->launchParams.probeHeightRes = scene.grid.resolution.y;

        // Radiance cell data
        radianceCellScatterCubeMapPipeline->launchParams.cellCenter = scene.grid.getCell(12).getCenter();
        radianceCellScatterCubeMapPipeline->launchParams.cellSize = scene.grid.getCellSize();

        // UV world position data
        //radianceCellScatterCubeMapPipeline->launchParams.uvWorldPositions.size = texSize;
        //radianceCellScatterCubeMapPipeline->launchParams.uvWorldPositions.UVDataBuffer = (UVWorldData*)samplePointWorldPositionDeviceBuffer.d_pointer();

        radianceCellScatterCubeMapPipeline->uploadLaunchParams();

        OPTIX_CHECK(optixLaunch(
            radianceCellScatterCubeMapPipeline->pipeline, stream,
            radianceCellScatterCubeMapPipeline->launchParamsBuffer.d_pointer(),
            radianceCellScatterCubeMapPipeline->launchParamsBuffer.sizeInBytes,
            &radianceCellScatterCubeMapPipeline->sbt,
            1,                     // dimension X: amount of UV texels in the cell
            1,                     // dimension Y: 1
            1                      // dimension Z: 1
            // dimension X * dimension Y * dimension Z CUDA threads will be spawned 
        ));

        CUDA_SYNC_CHECK();
    }

    void Renderer::octreeTextureTest()
    {
        std::cout << "OCTREE TESTING... " << std::endl;

        //radianceCellScatterCubeMapPipeline->launchParams.octreeTexture = (float*)octreeTextures->getOctreeGPUMemory().d_pointer();

        radianceCellScatterCubeMapPipeline->uploadLaunchParams();

        OPTIX_CHECK(optixLaunch(
            radianceCellScatterCubeMapPipeline->pipeline, stream,
            radianceCellScatterCubeMapPipeline->launchParamsBuffer.d_pointer(),
            radianceCellScatterCubeMapPipeline->launchParamsBuffer.sizeInBytes,
            &radianceCellScatterCubeMapPipeline->sbt,
            1,                     // dimension X: amount of UV texels in the cell
            1,                     // dimension Y: 1
            1                      // dimension Z: 1
            // dimension X * dimension Y * dimension Z CUDA threads will be spawned 
        ));

        CUDA_SYNC_CHECK();
    }

    void Renderer::textureAndSurfaceObjectTest()
    {
        std::cout << "TESTING TEXTURE AND SURFACE OBJECTS... " << std::endl;


        //radianceCellScatterCubeMapPipeline->launchParams.textureRef = textureObjects[0];
        //radianceCellScatterCubeMapPipeline->launchParams.surfaceRef = surfaceObjects[0];
        //radianceCellScatterCubeMapPipeline->launchParams.read = false;
        radianceCellScatterCubeMapPipeline->uploadLaunchParams();

        // Surface write
        OPTIX_CHECK(optixLaunch(
            radianceCellScatterCubeMapPipeline->pipeline, stream,
            radianceCellScatterCubeMapPipeline->launchParamsBuffer.d_pointer(),
            radianceCellScatterCubeMapPipeline->launchParamsBuffer.sizeInBytes,
            &radianceCellScatterCubeMapPipeline->sbt,
            1,                     // dimension X: amount of UV texels in the cell
            1,                     // dimension Y: 1
            1                      // dimension Z: 1
            // dimension X * dimension Y * dimension Z CUDA threads will be spawned 
        ));

        CUDA_SYNC_CHECK();


        // Texture read
        //radianceCellScatterCubeMapPipeline->launchParams.read = true;
        radianceCellScatterCubeMapPipeline->uploadLaunchParams();
        OPTIX_CHECK(optixLaunch(
            radianceCellScatterCubeMapPipeline->pipeline, stream,
            radianceCellScatterCubeMapPipeline->launchParamsBuffer.d_pointer(),
            radianceCellScatterCubeMapPipeline->launchParamsBuffer.sizeInBytes,
            &radianceCellScatterCubeMapPipeline->sbt,
            1,                     // dimension X: amount of UV texels in the cell
            1,                     // dimension Y: 1
            1                      // dimension Z: 1
            // dimension X * dimension Y * dimension Z CUDA threads will be spawned 
        ));

        CUDA_SYNC_CHECK();
    }



    void Renderer::calculateRadianceCellScatterUnbiased(int iteration, cudaTextureObject_t* prevBounceTexture, cudaSurfaceObject_t* dstTexture)
    {
        // TODO: For now we're using the same texture size as for the direct lighting pass, we can downsample in the future to gain performance
        const int texSize = directTextureSizes[0];
        NonEmptyCells nonEmpties = scene.grid.getNonEmptyCells();

        for (int i = 0; i < nonEmpties.nonEmptyCells.size(); i++)
        {
            radianceCellScatterUnbiasedPipeline->launchParams.nonEmptyCellIndex = i;

            // Light source texture data
            radianceCellScatterUnbiasedPipeline->launchParams.prevBounceTextures = prevBounceTexture;

            // Destination texture data
            radianceCellScatterUnbiasedPipeline->launchParams.currentBounceTextures = dstTexture;

            // Load uvs per cell 
            std::vector<glm::vec2> cellUVs = nonEmpties.nonEmptyCells[i]->getUVsInside();
            std::cout << "Non-empty cell " << i << ": UVs in this cell: " << cellUVs.size() << std::endl;

            // Radiance cell data
            radianceCellScatterUnbiasedPipeline->launchParams.cellCenter = nonEmpties.nonEmptyCells[i]->getCenter();
            radianceCellScatterUnbiasedPipeline->launchParams.cellSize = scene.grid.getCellSize();

            // UV world position data
            radianceCellScatterUnbiasedPipeline->launchParams.uvPositions = (cudaTextureObject_t*)uvWorldPositionTextureObjectPointersBuffer.d_pointer();
            radianceCellScatterUnbiasedPipeline->launchParams.uvNormals = (cudaTextureObject_t*)uvNormalTextureObjectPointersBuffer.d_pointer();
            radianceCellScatterUnbiasedPipeline->launchParams.uvDiffuseColors = (cudaTextureObject_t*)uvDiffuseColorTextureObjectPointersBuffer.d_pointer();
            radianceCellScatterUnbiasedPipeline->launchParams.objectTextureResolutions = (int*)textureSizesBuffer.d_pointer();

            // Diffuse texture data
            radianceCellScatterUnbiasedPipeline->launchParams.hasTexture = (int*)objectHasTextureBuffer.d_pointer();
            radianceCellScatterUnbiasedPipeline->launchParams.diffuseTextures = (cudaTextureObject_t*)diffuseTextureObjectPointersBuffer.d_pointer();
            radianceCellScatterUnbiasedPipeline->launchParams.diffuseTextureUVs = (cudaTextureObject_t*)diffuseTextureUVsTextureObjectPointersBuffer.d_pointer();

            radianceCellScatterUnbiasedPipeline->uploadLaunchParams();

            OPTIX_CHECK(optixLaunch(
                radianceCellScatterUnbiasedPipeline->pipeline, stream,
                radianceCellScatterUnbiasedPipeline->launchParamsBuffer.d_pointer(),
                radianceCellScatterUnbiasedPipeline->launchParamsBuffer.sizeInBytes,
                &radianceCellScatterUnbiasedPipeline->sbt,
                cellUVs.size(),                                        // dimension X: amount of UV texels in the cell
                1,                                                     // dimension Y: 1
                1                                                      // dimension Z: 1
            ));

            CUDA_SYNC_CHECK();
        }

        //// Download resulting texture from GPU
        //std::vector<uint32_t> current_bounce_result(radianceCellScatterUnbiasedPipeline->launchParams.currentBounceTexture.size * radianceCellScatterUnbiasedPipeline->launchParams.currentBounceTexture.size);
        //dstTexture.download(current_bounce_result.data(),
        //    radianceCellScatterUnbiasedPipeline->launchParams.currentBounceTexture.size * radianceCellScatterUnbiasedPipeline->launchParams.currentBounceTexture.size);

        //// Write the result to an image (for debugging purposes)
        //writeToImage("current_bounce_output" + std::to_string(iteration) + ".png", radianceCellScatterUnbiasedPipeline->launchParams.currentBounceTexture.size, radianceCellScatterUnbiasedPipeline->launchParams.currentBounceTexture.size, current_bounce_result.data());
    }

    void Renderer::calculateDirectLightingOctree()
    {
        std::cout << "=======================================================" << std::endl;
        std::cout << "      CALCULATE DIRECT LIGHTING (OCTREE TEXTURE)       " << std::endl;
        std::cout << "=======================================================" << std::endl;

        const int texSize = IRRADIANCE_TEXTURE_RESOLUTION;

        // Get lights data from scene
        std::vector<LightData> lightData = scene.getLightsData();

        // Allocate device space for the light data buffer, then upload the light data to the device
        lightDataBuffer.resize(lightData.size() * sizeof(LightData));
        lightDataBuffer.upload(lightData.data(), 1);

        directLightPipelineOctree->launchParams.amountLights = lightData.size();
        directLightPipelineOctree->launchParams.lights = (LightData*)lightDataBuffer.d_pointer();
        directLightPipelineOctree->launchParams.stratifyResX = STRATIFIED_X_SIZE;
        directLightPipelineOctree->launchParams.stratifyResY = STRATIFIED_Y_SIZE;

        // Octree data
        int granularity = octreeTextures->getKernelGranularity();
        directLightPipelineOctree->launchParams.granularity = granularity;
        directLightPipelineOctree->launchParams.octreeTexture = (float*)octreeTextures->getOctreeGPUMemory().d_pointer();
        directLightPipelineOctree->launchParams.UVWorldPosTextureResolution = texSize;

        directLightPipelineOctree->uploadLaunchParams();

        // Launch direct lighting pipeline
        OPTIX_CHECK(optixLaunch(
            directLightPipelineOctree->pipeline, stream,
            directLightPipelineOctree->launchParamsBuffer.d_pointer(),
            directLightPipelineOctree->launchParamsBuffer.sizeInBytes,
            &directLightPipelineOctree->sbt,
            directLightPipelineOctree->launchParams.uvWorldPositions.size,                    // dimension X: texture size (UV world positions)
            1,                                                                                // dimension Y: texture size (UV world positions)
            1                                                                                 // dimension Z: 1
            // dimension X * dimension Y * dimension Z CUDA threads will be spawned 
        ));

        CUDA_SYNC_CHECK();
    }

    void Renderer::calculateIndirectLightingOctree(BIAS_MODE bias, PROBE_MODE mode)
    {
        if (bias == BIASED_PROBES)
        {
            if (mode == SPHERICAL_HARMONICS)
            {
                std::cout << "========================================================" << std::endl;
                std::cout << "== TODO == INDIRECT LIGHTING CALCULATION (SH) == TODO == " << std::endl;
                std::cout << "========================================================" << std::endl;

                // =======
                // TODO
                // =======
               /* for (int i = 0; i < 1; i++)
                {
                    std::cout << "Calculating radiance cell gather pass " << i << "..." << std::endl;

                    switch (i)
                    {
                    case 0:
                        calculateRadianceCellGatherPass(directLightingTexture);
                        std::cout << "Calculating radiance cell scatter pass " << i << "..." << std::endl;
                        calculateRadianceCellScatterPass(i, secondBounceTexture);
                        break;
                    case 1:
                        calculateRadianceCellGatherPass(secondBounceTexture);
                        std::cout << "Calculating radiance cell scatter pass " << i << "..." << std::endl;
                        calculateRadianceCellScatterPass(i, thirdBounceTexture);
                        break;
                    default:
                        break;
                    }
                }*/
            }
            else if (mode == CUBE_MAP)
            {
                std::cout << "====================================================" << std::endl;
                std::cout << "      INDIRECT LIGHTING CALCULATION (CUBEMAP)       " << std::endl;
                std::cout << "====================================================" << std::endl;

                for (int i = 0; i < 1; i++)
                {
                    std::cout << "Calculating radiance cell gather pass " << i << "..." << std::endl;

                    switch (i)
                    {
                    case 0:
                        calculateRadianceCellGatherPassCubeMapAltOctree(octreeTextures->getOctreeGPUMemory());
                        std::cout << "Calculating radiance cell scatter pass " << i << "..." << std::endl;
                        calculateRadianceCellScatterPassCubeMapOctree(i, octreeTextures->getOctreeGPUMemory(), octreeTextures->getOctreeGPUMemoryBounce2());
                        break;
                    case 1:
    /*                    calculateRadianceCellGatherPassCubeMapAltOctree(secondBounceTexture);
                        std::cout << "Calculating radiance cell scatter pass " << i << "..." << std::endl;
                        calculateRadianceCellScatterPassCubeMapOctree(i, secondBounceTexture, thirdBounceTexture);*/
                        break;
                    default:
                        break;
                    }
                }
            }
        }
        else if (bias == UNBIASED)
        {
            std::cout << "==================================================" << std::endl;
            std::cout << "     INDIRECT LIGHTING CALCULATION (UNBIASED)     " << std::endl;
            std::cout << "==================================================" << std::endl;

            for (int i = 0; i < 2; i++)
            {
                std::cout << "Unbiased approach iteration " << i << "..." << std::endl;
                switch (i)
                {
                case 0:
                    calculateRadianceCellScatterUnbiasedOctree(i, octreeTextures->getOctreeGPUMemory(), octreeTextures->getOctreeGPUMemoryBounce2());
                    break;
                case 1:
  /*                  calculateRadianceCellScatterUnbiasedOctree(i, secondBounceTexture, thirdBounceTexture);*/
                    break;
                default:
                    break;
                }
            }
        }
    }

    void Renderer::calculateRadianceCellGatherPassCubeMapAltOctree(CUDABuffer& previousPassLightSourceOctreeTexture)
    {
        // TODO: For now we're using the same texture size as for the direct lighting pass, we can downsample in the future to gain performance
        const int texSize = IRRADIANCE_TEXTURE_RESOLUTION;
        const float cellSize = scene.grid.getCellSize();

        // Initialize Light Source Octree Texture data on GPU
        radianceCellGatherCubeMapPipelineOctree->launchParams.lightSourceOctreeTexture = (float*)previousPassLightSourceOctreeTexture.d_pointer();

        // Initialize UV World positions data on GPU
        radianceCellGatherCubeMapPipelineOctree->launchParams.uvWorldPositions.size = texSize * texSize;
        radianceCellGatherCubeMapPipelineOctree->launchParams.uvWorldPositions.UVDataBuffer = (UVWorldData*)samplePointWorldPositionDeviceBuffer.d_pointer();

        // Initialize cell size in launch params
        radianceCellGatherCubeMapPipelineOctree->launchParams.cellSize = cellSize;

        for (int z = 0; z < scene.grid.resolution.z; z++)
        {
            for (int y = 0; y < scene.grid.resolution.y; y++)
            {
                for (int x = 0; x < scene.grid.resolution.x; x++)
                {
                    int currentProbeOffset = ((z * scene.grid.resolution.x * scene.grid.resolution.y) + (y * scene.grid.resolution.x) + (x)) * 6 * radianceCellGatherCubeMapPipelineOctree->launchParams.cubeMapResolution * radianceCellGatherCubeMapPipelineOctree->launchParams.cubeMapResolution;
                    radianceCellGatherCubeMapPipelineOctree->launchParams.probeOffset = currentProbeOffset;
                    radianceCellGatherCubeMapPipelineOctree->launchParams.probePosition = glm::vec3{ x * cellSize + (0.5f * cellSize), y * cellSize + (0.5f * cellSize), z * cellSize + (0.5f * cellSize) }; // Probes are centered in each radiance cell
                    radianceCellGatherCubeMapPipelineOctree->uploadLaunchParams();

                    OPTIX_CHECK(optixLaunch(
                        radianceCellGatherCubeMapPipelineOctree->pipeline, stream,
                        radianceCellGatherCubeMapPipelineOctree->launchParamsBuffer.d_pointer(),
                        radianceCellGatherCubeMapPipelineOctree->launchParamsBuffer.sizeInBytes,
                        &radianceCellGatherCubeMapPipelineOctree->sbt,
                        radianceCellGatherCubeMapPipelineOctree->launchParams.cubeMapResolution,          // dimension X: divisionResX: width of cubemap face texture
                        radianceCellGatherCubeMapPipelineOctree->launchParams.cubeMapResolution,          // dimension Y: divisionResY: height of cubemap face texture
                        6                                                                                 // dimension Z: amount of cubemap faces
                        // dimension X * dimension Y * dimension Z CUDA threads will be spawned 
                    ));
                }
            }
        }
    }


    void Renderer::calculateRadianceCellScatterPassCubeMapOctree(int iteration, CUDABuffer& prevBounceOctreeTexture, CUDABuffer& dstOctreeTexture)
    {
        // TODO: For now we're using the same texture size as for the direct lighting pass, we can downsample in the future to gain performance
        const int texSize = IRRADIANCE_TEXTURE_RESOLUTION;
        NonEmptyCells nonEmpties = scene.grid.getNonEmptyCells();

        for (int i = 0; i < nonEmpties.nonEmptyCells.size(); i++)
        {
            radianceCellScatterCubeMapPipelineOctree->launchParams.nonEmptyCellIndex = i;
            radianceCellScatterCubeMapPipelineOctree->launchParams.cellCoords = nonEmpties.nonEmptyCells[i]->getCellCoords();
            radianceCellScatterCubeMapPipelineOctree->launchParams.probeWidthRes = scene.grid.resolution.x;
            radianceCellScatterCubeMapPipelineOctree->launchParams.probeHeightRes = scene.grid.resolution.y;

            // Texture that we write to
            radianceCellScatterCubeMapPipelineOctree->launchParams.currentBounceOctreeTexture = (float*)dstOctreeTexture.d_pointer();

            // Indirect light source texture that we read from (in case of local ray tracing)
            radianceCellScatterCubeMapPipelineOctree->launchParams.prevBounceOctreeTexture = (float*)prevBounceOctreeTexture.d_pointer();

            // Load uvs per cell 
            std::vector<glm::vec2> cellUVs = nonEmpties.nonEmptyCells[i]->getUVsInside();
            std::cout << "Iteration " << i << ": UVs in this cell: " << cellUVs.size() << std::endl;

            // Radiance cell data
            radianceCellScatterCubeMapPipelineOctree->launchParams.cellCenter = nonEmpties.nonEmptyCells[i]->getCenter();
            radianceCellScatterCubeMapPipelineOctree->launchParams.cellSize = scene.grid.getCellSize();

            // UV world position data
            radianceCellScatterCubeMapPipelineOctree->launchParams.uvWorldPositions.size = texSize;
            radianceCellScatterCubeMapPipelineOctree->launchParams.uvWorldPositions.UVDataBuffer = (UVWorldData*)samplePointWorldPositionDeviceBuffer.d_pointer();

            radianceCellScatterCubeMapPipelineOctree->uploadLaunchParams();

            OPTIX_CHECK(optixLaunch(
                radianceCellScatterCubeMapPipelineOctree->pipeline, stream,
                radianceCellScatterCubeMapPipelineOctree->launchParamsBuffer.d_pointer(),
                radianceCellScatterCubeMapPipelineOctree->launchParamsBuffer.sizeInBytes,
                &radianceCellScatterCubeMapPipelineOctree->sbt,
                cellUVs.size(),                     // dimension X: amount of UV texels in the cell
                1,                                  // dimension Y: 1
                1                                   // dimension Z: 1
                // dimension X * dimension Y * dimension Z CUDA threads will be spawned 
            ));

            CUDA_SYNC_CHECK();
        }
    }

    void Renderer::calculateRadianceCellScatterUnbiasedOctree(int iteration, CUDABuffer& prevBounceOctreeTexture, CUDABuffer& dstOctreeTexture)
    {
        // TODO: For now we're using the same texture size as for the direct lighting pass, we can downsample in the future to gain performance
        const int texSize = IRRADIANCE_TEXTURE_RESOLUTION;
        NonEmptyCells nonEmpties = scene.grid.getNonEmptyCells();

        for (int i = 0; i < nonEmpties.nonEmptyCells.size(); i++)
        {
            radianceCellScatterUnbiasedPipelineOctree->launchParams.nonEmptyCellIndex = i;

            // Light source texture data
            radianceCellScatterUnbiasedPipelineOctree->launchParams.prevBounceOctreeTexture = (float*)prevBounceOctreeTexture.d_pointer();

            // Destination texture data
            radianceCellScatterUnbiasedPipelineOctree->launchParams.currentBounceOctreeTexture = (float*)dstOctreeTexture.d_pointer();

            // Load uvs per cell 
            std::vector<glm::vec2> cellUVs = nonEmpties.nonEmptyCells[i]->getUVsInside();
            std::cout << "Non-empty cell " << i << ": UVs in this cell: " << cellUVs.size() << std::endl;

            // Radiance cell data
            radianceCellScatterUnbiasedPipelineOctree->launchParams.cellCenter = nonEmpties.nonEmptyCells[i]->getCenter();
            radianceCellScatterUnbiasedPipelineOctree->launchParams.cellSize = scene.grid.getCellSize();

            // UV world position data
            radianceCellScatterUnbiasedPipelineOctree->launchParams.uvWorldPositions.size = texSize;
            radianceCellScatterUnbiasedPipelineOctree->launchParams.uvWorldPositions.UVDataBuffer = (UVWorldData*)samplePointWorldPositionDeviceBuffer.d_pointer();

            radianceCellScatterUnbiasedPipelineOctree->uploadLaunchParams();

            OPTIX_CHECK(optixLaunch(
                radianceCellScatterUnbiasedPipelineOctree->pipeline, stream,
                radianceCellScatterUnbiasedPipelineOctree->launchParamsBuffer.d_pointer(),
                radianceCellScatterUnbiasedPipelineOctree->launchParamsBuffer.sizeInBytes,
                &radianceCellScatterUnbiasedPipelineOctree->sbt,
                cellUVs.size(),                                        // dimension X: amount of UV texels in the cell
                1,                                                     // dimension Y: 1
                1                                                      // dimension Z: 1
            ));

            CUDA_SYNC_CHECK();
        }
    }



    void Renderer::loadLightTexture()
    {
        int width;
        int height;
        int comp;
        //stbi_uc* pixels = stbi_load("../textures/cornell_uv_light.png", &width, &height, &comp, 4);

        stbi_uc* pixels = stbi_load("../textures/cornell_uv_light.png", &width, &height, &comp, 4);

        std::cout << width << std::endl;
        std::cout << height << std::endl;
        std::cout << comp << std::endl;
        if (pixels == NULL) {
            std::cout << "Error while loading light texture!" << std::endl;
            return;
        }

        // Upload image to GPU
        lightSourceTexture.alloc(width * height * comp * sizeof(stbi_uc));
        lightSourceTexture.upload(pixels, width * height * comp);

        stbi_image_free(pixels);
    }


    void Renderer::writeWeightsToTxtFile(std::vector<float>& weights, std::vector<int>& numSamples, int amountCells)
    {
        std::ofstream outputFile;
        outputFile.open("../weights.txt");


        for (int i = 0; i < amountCells; i++)
        {
            int cellOffset = i * 8 * SPHERICAL_HARMONIC_BASIS_FUNCTIONS;
            for (int sh = 0; sh < 8; sh++)
            {
                float weight;
                if (numSamples[i * 8 + sh] > 0)
                {
                    weight = 1.0f / (numSamples[i * 8 + sh] * 4 * glm::pi<float>());
                }
                else {
                    weight = 0.0f;
                }
                int shOffset = sh * SPHERICAL_HARMONIC_BASIS_FUNCTIONS;
                for (int bf = 0; bf < 9; bf++)
                {   
                    if (bf < 8)
                    {
                        outputFile << weights[cellOffset + shOffset + bf] * weight << " ";
                    }
                    else {
                        outputFile << weights[cellOffset + shOffset + bf] * weight;
                    }
                }
                outputFile << "\n";
            }
        }
    }

    void Renderer::prepareUVWorldPositionsPerObject(WORLD_DATA_LOAD loadOrCalculate)
    {
        std::string saveDirPositions = "../data/world_data_textures/positions/";
        std::string saveDirNormals = "../data/world_data_textures/normals/";
        std::string saveDirDiffuseCoords = "../data/world_data_textures/diffuse_coords/";

        std::vector<std::string> fileNames {"arcs.txt", "arcs2.txt", "building_core.txt", "bushes.txt", "curtains.txt", 
        "curtains2.txt", "curtains3.txt","doors.txt","drapes.txt", "drapes2.txt", "drapes3.txt", "fire_pit.txt", "lion.txt", 
        "lion_background.txt", "pillars.txt", "pillars_up.txt", "pillars_up2.txt", "plants.txt", "platforms.txt", "pots.txt",
        "roof.txt", "spears.txt", "spikes.txt", "square_panel_back.txt", "water_dish.txt"};

        int numObjects = scene.getGameObjects().size();

        UVWorldPositionsTextures.resize(numObjects);
        UVNormalsTextures.resize(numObjects);
        UVDiffuseColorTextures.resize(numObjects);
        diffuseTextureUVsTextures.resize(numObjects);
        objectHasTexture.resize(numObjects);

        for (int o = 0; o < numObjects; o++)
        {
            std::cout << "SANITY CHECK: amount of vertices " << scene.getGameObjects()[o]->model->mesh->vertices.size() << ", amount of diffuse UVs: " << scene.getGameObjects()[o]->model->mesh->diffuseTextureCoords.size() << std::endl;

            objectHasTexture[o] = scene.getGameObjects()[o]->model->mesh->diffuseTextureID > -1;

            int texSize = directTextureSizes[o];

            std::vector<float> worldPositions;
            std::vector<float> worldNormals;
            std::vector<float> diffuseColors;
            std::vector<float> diffuseTextureUVs;

            if (loadOrCalculate == LOAD_WORLD_DATA)
            {
                diffuseColors.resize(texSize * texSize * 4, 0.0f);                

                readWorldDataTextureFromFile(saveDirPositions + fileNames[o], worldPositions);
                readWorldDataTextureFromFile(saveDirNormals + fileNames[o], worldNormals);
                readWorldDataTextureFromFile(saveDirDiffuseCoords + fileNames[o], diffuseTextureUVs);

                // Assign UVs to radiance grid cells
                for (int i = 0; i < texSize * texSize; i++)
                {
                    const float u = float(i % texSize) / float(texSize);
                    // (i - i % texSize) / texSize gives us the row number, divided by texSize gives us the V coordinate 
                    const float v = (float((i - (i % texSize))) / float(texSize)) / float(texSize);
                    glm::vec2 uv = glm::vec2{ u,v };
                    
                    glm::vec3 worldPosition = glm::vec3{ worldPositions[i * 4 + 0], worldPositions[i * 4 + 1], worldPositions[i * 4 + 2] };

                    // If this uv coordinate actually contains a surface (and therefore changed the initialization value)
                    if (worldPosition.x != -1000.0f)
                    {
                        scene.grid.assignUVToCells(uv, worldPosition, o);
                    }
                }
            }
            else if (loadOrCalculate == CALCULATE_WORLD_DATA)
            {
                worldPositions.clear();
                worldNormals.clear();
                diffuseColors.clear();
                diffuseTextureUVs.clear();
                worldPositions.resize(texSize * texSize * 4, -1000.0f);
                worldNormals.resize(texSize * texSize * 4, 0.0f);
                diffuseColors.resize(texSize * texSize * 4, 0.0f);
                diffuseTextureUVs.resize(texSize * texSize * 4, 0.0f);

                float progress = 0.1f;
                for (int i = 0; i < texSize * texSize; i++)
                {
                    const float u = float(i % texSize) / float(texSize);
                    // (i - i % texSize) / texSize gives us the row number, divided by texSize gives us the V coordinate 
                    const float v = (float((i - (i % texSize))) / float(texSize)) / float(texSize);
                    glm::vec2 uv = glm::vec2{ u,v };
                    UVWorldData uvWorldData = UVto3DPerObject(uv, scene.getGameObjects()[o]);

                    worldPositions[i * 4 + 0] = uvWorldData.worldPosition.x;
                    worldPositions[i * 4 + 1] = uvWorldData.worldPosition.y;
                    worldPositions[i * 4 + 2] = uvWorldData.worldPosition.z;
                    worldPositions[i * 4 + 3] = 0.0f;

                    worldNormals[i * 4 + 0] = uvWorldData.worldNormal.x;
                    worldNormals[i * 4 + 1] = uvWorldData.worldNormal.y;
                    worldNormals[i * 4 + 2] = uvWorldData.worldNormal.z;
                    worldNormals[i * 4 + 3] = 0.0f;

                    diffuseColors[i * 4 + 0] = uvWorldData.diffuseColor.x;
                    diffuseColors[i * 4 + 1] = uvWorldData.diffuseColor.y;
                    diffuseColors[i * 4 + 2] = uvWorldData.diffuseColor.z;
                    diffuseColors[i * 4 + 3] = 0.0f;

                    diffuseTextureUVs[i * 4 + 0] = uvWorldData.diffuseTextureUV.x;
                    diffuseTextureUVs[i * 4 + 1] = uvWorldData.diffuseTextureUV.y;
                    diffuseTextureUVs[i * 4 + 2] = 0.0f;
                    diffuseTextureUVs[i * 4 + 3] = 0.0f;

                    // Assign this UV to the cell that it belongs to, so in the scattering pass we can operate on local UVs
                    scene.grid.assignUVToCells(uv, uvWorldData.worldPosition, o);

                    if (float(i) / float(texSize * texSize) > progress)
                    {
                        std::cout << "Progress: " << progress * 100 << "%." << std::endl;
                        progress += 0.1f;
                    }
                }

 /*               writeWorldDataTextureToFile(worldPositions, saveDirPositions + fileNames[o]);
                writeWorldDataTextureToFile(worldNormals, saveDirNormals + fileNames[o]);
                writeWorldDataTextureToFile(diffuseTextureUVs, saveDirDiffuseCoords + fileNames[o]);*/
            }

  /*          for (int i = 0; i < worldPositions.size(); i++)
            {
                std::cout << worldPositions[i] << std::endl;
            }*/
            
            //HDRImage hdr{ texSize , texSize, worldPositions.data() };
            //hdr.saveImage("../debug_output/worldpos.hdr");

            //HDRImage hdrNormals{ texSize , texSize, worldNormals.data() };
            //hdrNormals.saveImage("../debug_output/worldnormals.hdr");

            //HDRImage hdrDiffCoord{ texSize , texSize, diffuseTextureUVs.data() };
            //hdrDiffCoord.saveImage("../debug_output/diffcoords.hdr");

            cudaChannelFormatDesc channelDesc;
            channelDesc = cudaCreateChannelDesc<float4>();
            int32_t width = texSize;
            int32_t height = texSize;
            int32_t pitch = width * sizeof(float4);

            // Initialize and allocate CUDA arrays
            cudaArray* worldPosArray;
            cudaArray* worldNormalArray;
            cudaArray* diffuseColorArray;
            cudaArray* diffuseTextureUVArray;


            CUDA_CHECK(MallocArray(&worldPosArray,
                &channelDesc,
                width,
                height,
                cudaArraySurfaceLoadStore));

            CUDA_CHECK(MallocArray(&worldNormalArray,
                &channelDesc,
                width,
                height,
                cudaArraySurfaceLoadStore));

            CUDA_CHECK(MallocArray(&diffuseColorArray,
                &channelDesc,
                width,
                height,
                cudaArraySurfaceLoadStore));

            CUDA_CHECK(MallocArray(&diffuseTextureUVArray,
                &channelDesc,
                width,
                height,
                cudaArraySurfaceLoadStore));

            CUDA_CHECK(Memcpy2DToArray(worldPosArray,
                /* offset */0, 0,
                worldPositions.data(),
                pitch, pitch, height,
                cudaMemcpyHostToDevice));


            CUDA_CHECK(Memcpy2DToArray(worldNormalArray,
                /* offset */0, 0,
                worldNormals.data(),
                pitch, pitch, height,
                cudaMemcpyHostToDevice));


            CUDA_CHECK(Memcpy2DToArray(diffuseColorArray,
                /* offset */0, 0,
                diffuseColors.data(),
                pitch, pitch, height,
                cudaMemcpyHostToDevice));

            CUDA_CHECK(Memcpy2DToArray(diffuseTextureUVArray,
                /* offset */0, 0,
                diffuseTextureUVs.data(),
                pitch, pitch, height,
                cudaMemcpyHostToDevice));

            // Resource description for the CUDA arrays
            cudaResourceDesc resourceDescPositions = {};
            resourceDescPositions.resType = cudaResourceTypeArray;
            resourceDescPositions.res.array.array = worldPosArray;

            cudaResourceDesc resourceDescNormals = {};
            resourceDescNormals.resType = cudaResourceTypeArray;
            resourceDescNormals.res.array.array = worldNormalArray;

            cudaResourceDesc resourceDescColors = {};
            resourceDescColors.resType = cudaResourceTypeArray;
            resourceDescColors.res.array.array = diffuseColorArray;

            cudaResourceDesc resourceDescDiffuseUVs = {};
            resourceDescDiffuseUVs.resType = cudaResourceTypeArray;
            resourceDescDiffuseUVs.res.array.array = diffuseTextureUVArray;

            // Texture object creation
            cudaTextureDesc texDesc = {};
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModePoint;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 1;
            texDesc.maxAnisotropy = 1;
            texDesc.maxMipmapLevelClamp = 99;
            texDesc.minMipmapLevelClamp = 0;
            texDesc.mipmapFilterMode = cudaFilterModePoint;
            texDesc.borderColor[0] = 1.0f;
            texDesc.sRGB = 0;

            // Create texture objects
            cudaTextureObject_t cudaTexPositions = 0;
            CUDA_CHECK(CreateTextureObject(&cudaTexPositions, &resourceDescPositions, &texDesc, nullptr));
            UVWorldPositionsTextures[o] = cudaTexPositions;

            cudaTextureObject_t cudaTexNormals = 0;
            CUDA_CHECK(CreateTextureObject(&cudaTexNormals, &resourceDescNormals, &texDesc, nullptr));
            UVNormalsTextures[o] = cudaTexNormals;

            cudaTextureObject_t cudaTexDiffuseColors = 0;
            CUDA_CHECK(CreateTextureObject(&cudaTexDiffuseColors, &resourceDescColors, &texDesc, nullptr));
            UVDiffuseColorTextures[o] = cudaTexDiffuseColors;

            cudaTextureObject_t cudaTexDiffuseTextureUVs = 0;
            CUDA_CHECK(CreateTextureObject(&cudaTexDiffuseTextureUVs, &resourceDescDiffuseUVs, &texDesc, nullptr));
            diffuseTextureUVsTextures[o] = cudaTexDiffuseTextureUVs;
        }

        // Buffers to store pointers to texture objects
        uvWorldPositionTextureObjectPointersBuffer.alloc_and_upload(UVWorldPositionsTextures);
        uvNormalTextureObjectPointersBuffer.alloc_and_upload(UVNormalsTextures);
        uvDiffuseColorTextureObjectPointersBuffer.alloc_and_upload(UVDiffuseColorTextures);
        diffuseTextureUVsTextureObjectPointersBuffer.alloc_and_upload(diffuseTextureUVsTextures);
        objectHasTextureBuffer.alloc_and_upload(objectHasTexture);
    }


    // Note that this function must be called after "prepareUVWorldPositions", because the UVs have to be assigned to each cell first
    void Renderer::prepareUVsInsideBuffer()
    {
        NonEmptyCells nonEmpties = scene.grid.getNonEmptyCells();

        std::vector<int> offsets;
        std::vector<glm::vec2> cellUVs;
        std::vector<int> uvGameObjectNrs;

        
        for (int i = 0; i < nonEmpties.nonEmptyCells.size(); i++)
        {
            // Load uvs per cell 
            offsets.push_back(cellUVs.size());
            cellUVs.insert(cellUVs.end(), nonEmpties.nonEmptyCells[i]->getUVsInside().begin(), nonEmpties.nonEmptyCells[i]->getUVsInside().end());
            uvGameObjectNrs.insert(uvGameObjectNrs.end(), nonEmpties.nonEmptyCells[i]->getUVsGameObjectNrs().begin(), nonEmpties.nonEmptyCells[i]->getUVsGameObjectNrs().end());
        }

        // For debugging purposes and visualization only (should be commented out in release)
        // writeUVsPerCellToImage(offsets, cellUVs, 1024);

        UVsInsideBuffer.alloc_and_upload(cellUVs);
        UVsInsideOffsets.alloc_and_upload(offsets);
        UVsGameObjectNrsBuffer.alloc_and_upload(uvGameObjectNrs);

        if (irradStorageType == OCTREE_TEXTURE)
        {
            if (radianceCellScatterCubeMapPipelineOctree != nullptr)
            {
                radianceCellScatterCubeMapPipelineOctree->launchParams.uvsInside = (glm::vec2*)UVsInsideBuffer.d_pointer();
                radianceCellScatterCubeMapPipelineOctree->launchParams.uvsInsideOffsets = (int*)UVsInsideOffsets.d_pointer();
            }

            if (radianceCellScatterUnbiasedPipelineOctree != nullptr)
            {
                radianceCellScatterUnbiasedPipelineOctree->launchParams.uvsInside = (glm::vec2*)UVsInsideBuffer.d_pointer();
                radianceCellScatterUnbiasedPipelineOctree->launchParams.uvsInsideOffsets = (int*)UVsInsideOffsets.d_pointer();
            }
        }
        else if (irradStorageType == TEXTURE_2D)
        {
            if (radianceCellScatterPipeline != nullptr)
            {
                radianceCellScatterPipeline->launchParams.uvsInside = (glm::vec2*)UVsInsideBuffer.d_pointer();
                radianceCellScatterPipeline->launchParams.uvsInsideOffsets = (int*)UVsInsideOffsets.d_pointer();
            }

            if (radianceCellScatterCubeMapPipeline != nullptr)
            {
                radianceCellScatterCubeMapPipeline->launchParams.uvsInside = (glm::vec2*)UVsInsideBuffer.d_pointer();
                radianceCellScatterCubeMapPipeline->launchParams.uvsInsideOffsets = (int*)UVsInsideOffsets.d_pointer();
                radianceCellScatterCubeMapPipeline->launchParams.uvGameObjectNrs = (int*)UVsGameObjectNrsBuffer.d_pointer();
            }

            if (radianceCellScatterUnbiasedPipeline != nullptr)
            {
                radianceCellScatterUnbiasedPipeline->launchParams.uvsInside = (glm::vec2*)UVsInsideBuffer.d_pointer();
                radianceCellScatterUnbiasedPipeline->launchParams.uvsInsideOffsets = (int*)UVsInsideOffsets.d_pointer();
                radianceCellScatterUnbiasedPipeline->launchParams.uvGameObjectNrs = (int*)UVsGameObjectNrsBuffer.d_pointer();
            }
        }
    }

    void Renderer::prepareWorldSamplePoints(float octreeLeafFaceArea)
    {
        std::vector<WorldSamplePointData> samplePointWorldData;

        // Loop through all game objects in the scene
        for (auto& g : scene.getGameObjects())
        {
            // Loop through all triangles
            for (auto& triangle : g->model->mesh->indices)
            {
                glm::vec3 v1 = g->model->mesh->vertices[triangle[0]];
                glm::vec3 v2 = g->model->mesh->vertices[triangle[1]];
                glm::vec3 v3 = g->model->mesh->vertices[triangle[2]];

                v1 = g->worldTransform.object2World * glm::vec4{ v1, 1.0f };
                v2 = g->worldTransform.object2World * glm::vec4{ v2, 1.0f };
                v3 = g->worldTransform.object2World * glm::vec4{ v3, 1.0f };

                float u,v,w;
             
                float triangleSurfaceArea = GeometryUtils::triangleArea3D(v1, v2, v3);
                int amountOfPointsToGenerate = 2 * std::ceil(triangleSurfaceArea / octreeLeafFaceArea);

                for (int i = 0; i < amountOfPointsToGenerate; i++)
                {
                    const int range_from = 0.0f;
                    const int range_to = 1.0f;
                    std::random_device                      rand_dev;
                    std::mt19937                            generator(rand_dev());
                    std::uniform_real_distribution<float>   distr(range_from, range_to);

                    float u0 = distr(generator);
                    float u1 = distr(generator);

                    float sqrtU0 = sqrtf(u0);

                    glm::vec3 uniformRandomPointInTriangle = (1- sqrtU0) * v1 + (sqrtU0 *(1 - u1)) * v2 + (u1 * sqrtU0) * v3;

                    barycentricCoordinates(uniformRandomPointInTriangle, v1, v2, v3, u, v, w);

                    WorldSamplePointData newSamplePoint;
                    newSamplePoint.worldPosition = uniformRandomPointInTriangle;
                    // Barycentric interpolated shading normal (alternative: geometric normal from triangle) and barycentric interpolated UV coords
                    newSamplePoint.worldNormal = glm::normalize(w * g->model->mesh->normals[triangle[0]] + u * g->model->mesh->normals[triangle[1]] + v * g->model->mesh->normals[triangle[2]]);
                    newSamplePoint.diffuseColor = g->model->mesh->diffuse;
                    newSamplePoint.hasTexture = false;

                    // Texture data
                    if (g->model->mesh->diffuseTextureID >= 0)
                    {   
                        newSamplePoint.hasTexture = true;
                        newSamplePoint.textureObject = diffuseTextureObjects[g->model->mesh->diffuseTextureID];
                        newSamplePoint.uvCoords = w * g->model->mesh->texCoords[triangle[0]] + u * g->model->mesh->texCoords[triangle[1]] + v * g->model->mesh->texCoords[triangle[2]];
                    }
                    
                    samplePointWorldData.push_back(newSamplePoint);
                }
            }
        }
        // Upload world positions to the GPU and pass a pointer to this memory into the launch params
        samplePointWorldPositionDeviceBuffer.alloc_and_upload(samplePointWorldData);

        if (irradStorageType == OCTREE_TEXTURE)
        {
            directLightPipelineOctree->launchParams.uvWorldPositions.size = samplePointWorldData.size();
            directLightPipelineOctree->launchParams.uvWorldPositions.UVDataBuffer = (WorldSamplePointData*)samplePointWorldPositionDeviceBuffer.d_pointer();
        }

        std::cout << "Done generating world sample points. Amount of points generated: " << samplePointWorldData.size() << std::endl;
    }

    void Renderer::writeWorldDataTextureToFile(std::vector<float>& worldDataTexture, std::string fileName)
    {
        std::ofstream outputFile(fileName);
        for (auto& e : worldDataTexture)
        {
            outputFile << e << std::endl;
        }
        outputFile.close();
        std::cout << "Saved " << fileName << " to file." << std::endl;
    }

    void Renderer::readWorldDataTextureFromFile(std::string fileName, std::vector<float>& container)
    {
        // Create a text string, which is used to output the text file
        std::string line;

        // Read from the text file
        std::ifstream inputFile(fileName);

        // Use a while loop together with the getline() function to read the file line by line
        while (getline(inputFile, line)) {
            
            container.push_back(std::stof(line));
        }

        // Close the file
        inputFile.close();
        std::cout << "Read " << fileName << " from file." << std::endl;

    }


    void Renderer::writeUVsPerCellToImage(std::vector<int>& offsets, std::vector<glm::vec2>& uvs, int texRes)
    {
        Image outputImage{texRes, texRes};
        std::vector<glm::vec3> randomColors;

        // Generate an amount of random colors so each cell has its own color
        for (int i = 0; i < offsets.size() - 1; i++)
        {
            randomColors.push_back(generateRandomColor());
        }

        for (int o = 0; o < offsets.size() - 1; o++)
        {
            for (int uv = offsets[o]; uv < offsets[o + 1]; uv++)
            {
               glm::vec2 uvCoords = uvs[uv];
               glm::ivec2 xyCoords = {glm::floor(uvCoords.x * texRes), glm::floor(uvCoords.y * texRes)};
               float fraction = float(o) / float(offsets.size());
               char color[3] = {randomColors[o].x * 255.0f, randomColors[o].y * 255.0f, randomColors[o].z * 255.0f };
               outputImage.writePixel(xyCoords.x, xyCoords.y, color);
            }
        }
        outputImage.flipY();
        outputImage.saveImage("../textures/uvs_per_cell.png");
    }


    void Renderer::barycentricCoordinates(glm::vec3 p, glm::vec3 a, glm::vec3 b, glm::vec3 c, float& u, float& v, float& w)
    {
        // ==============================================================================================================
        // https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
        // ==============================================================================================================
        glm::vec3 v0 = b - a, v1 = c - a, v2 = p - a;
        float d00 = glm::dot(v0, v0);
        float d01 = glm::dot(v0, v1);
        float d11 = glm::dot(v1, v1);
        float d20 = glm::dot(v2, v0);
        float d21 = glm::dot(v2, v1);
        float invDenom = 1.0f / (d00 * d11 - d01 * d01);
        v = (d11 * d20 - d01 * d21) * invDenom;
        w = (d00 * d21 - d01 * d20) * invDenom;
        u = 1.0f - v - w;
    }

    UVWorldData Renderer::UVto3D(glm::vec2 uv)
    {
        // Loop through all game objects in the scene
        for (auto& g : scene.getGameObjects())
        {
            // Loop through all triangles
            for (auto& triangle : g->model->mesh->indices)
            {
                // Get UV coordinates of triangle vertices
                glm::vec2 uv1 = g->model->mesh->texCoords[triangle[0]];
                glm::vec2 uv2 = g->model->mesh->texCoords[triangle[1]];
                glm::vec2 uv3 = g->model->mesh->texCoords[triangle[2]];

                // Barycentric interpolation to check whether our point is in the triangle
                glm::vec2 f1 = uv1 - uv;
                glm::vec2 f2 = uv2 - uv;
                glm::vec2 f3 = uv3 - uv;

                float a = GeometryUtils::triangleArea2D(uv1, uv2, uv3);
                if (a == 0.0f) continue;

                // Barycentric coordinates
                float a1 = GeometryUtils::triangleArea2D(uv2, uv3, uv) / a; if (a1 < 0) continue;
                float a2 = GeometryUtils::triangleArea2D(uv3, uv1, uv) / a; if (a2 < 0) continue;
                float a3 = GeometryUtils::triangleArea2D(uv1, uv2, uv) / a; if (a3 < 0) continue;

                //std::cout << "UV found!" << std::endl;
                glm::vec3 uvPosition = a1 * g->model->mesh->vertices[triangle[0]] + a2 * g->model->mesh->vertices[triangle[1]] + a3 * g->model->mesh->vertices[triangle[2]];
                glm::vec3 uvNormal = glm::normalize(a1 * g->model->mesh->normals[triangle[0]] + a2 * g->model->mesh->normals[triangle[1]] + a3 * g->model->mesh->normals[triangle[2]]);
                glm::vec3 diffuseColor = g->model->mesh->diffuse;

                uvPosition = g->worldTransform.object2World * glm::vec4{ uvPosition, 1.0f };
                return { uvPosition, uvNormal, diffuseColor};
            }
        }
        return { glm::vec3{ -1000.0f, -1000.0f, -1000.0f }, glm::vec3{ -1000.0f, -1000.0f, -1000.0f }, glm::vec3{0.0f, 0.0f, 0.0f} };
    }

    UVWorldData Renderer::UVto3DPerObject(glm::vec2 uv, std::shared_ptr<GameObject> o)
    {
        // Loop through all triangles
        for (auto& triangle : o->model->mesh->indices)
        {
            // Get UV coordinates of triangle vertices
            glm::vec2 uv1 = o->model->mesh->texCoords[triangle[0]];
            glm::vec2 uv2 = o->model->mesh->texCoords[triangle[1]];
            glm::vec2 uv3 = o->model->mesh->texCoords[triangle[2]];

            // Barycentric interpolation to check whether our point is in the triangle
            glm::vec2 f1 = uv1 - uv;
            glm::vec2 f2 = uv2 - uv;
            glm::vec2 f3 = uv3 - uv;

            float a = GeometryUtils::triangleArea2D(uv1, uv2, uv3);
            if (a == 0.0f) continue;

            // Barycentric coordinates
            float a1 = GeometryUtils::triangleArea2D(uv2, uv3, uv) / a; if (a1 < 0) continue;
            float a2 = GeometryUtils::triangleArea2D(uv3, uv1, uv) / a; if (a2 < 0) continue;
            float a3 = GeometryUtils::triangleArea2D(uv1, uv2, uv) / a; if (a3 < 0) continue;

            //std::cout << "UV found!" << std::endl;
            glm::vec3 uvPosition = a1 * o->model->mesh->vertices[triangle[0]] + a2 * o->model->mesh->vertices[triangle[1]] + a3 * o->model->mesh->vertices[triangle[2]];
            glm::vec3 uvNormal = glm::normalize(a1 * o->model->mesh->normals[triangle[0]] + a2 * o->model->mesh->normals[triangle[1]] + a3 * o->model->mesh->normals[triangle[2]]);
            glm::vec3 diffuseColor = o->model->mesh->diffuse;

            glm::vec2 diffuseTextureUV;
            if (!o->model->mesh->diffuseTextureCoords.empty())
            {
                diffuseTextureUV = a1 * o->model->mesh->diffuseTextureCoords[triangle[0]] + a2 * o->model->mesh->diffuseTextureCoords[triangle[1]] + a3 * o->model->mesh->diffuseTextureCoords[triangle[2]];
            }

            uvPosition = o->worldTransform.object2World * glm::vec4{ uvPosition, 1.0f };
            return { uvPosition, uvNormal, diffuseColor, diffuseTextureUV };
        }

        return{ glm::vec3{ -1000.0f, -1000.0f, -1000.0f }, glm::vec3{ -1000.0f, -1000.0f, -1000.0f }, glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec2{0.0f, 0.0f} };
    }


    void Renderer::resize(const glm::ivec2& newSize, GameObject* viewerObject)
    {
        // If window minimized
        if (newSize.x == 0 | newSize.y == 0) return;

        // Resize CUDA frame buffer
        colorBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));
    
        // Update launch parameters that are passed to OptiX launch
        if (irradStorageType == OCTREE_TEXTURE)
        {
            cameraPipelineOctree->launchParams.frame.size = newSize;
            cameraPipelineOctree->launchParams.frame.colorBuffer = (uint32_t*)colorBuffer.d_pointer();
        }
        else if (irradStorageType == TEXTURE_2D)
        {
            cameraPipeline->launchParams.frame.size = newSize;
            cameraPipeline->launchParams.frame.colorBuffer = (uint32_t*)colorBuffer.d_pointer();
        }

        // Reset camera, aspect may have changed
        updateCamera(viewerObject);
    }

    // Copy rendered color buffer from device to host memory for display
    void Renderer::downloadPixels(uint32_t h_pixels[])
    {
        if (irradStorageType == OCTREE_TEXTURE)
        {
            colorBuffer.download(h_pixels,
                cameraPipelineOctree->launchParams.frame.size.x * cameraPipelineOctree->launchParams.frame.size.y);
        } 
        else if (irradStorageType == TEXTURE_2D)
        {
            colorBuffer.download(h_pixels,
                cameraPipeline->launchParams.frame.size.x * cameraPipeline->launchParams.frame.size.y);
        }
    }
}