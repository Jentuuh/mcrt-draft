#include "renderer.hpp"

// This include may only appear in a SINGLE src file:
#include <optix_function_table_definition.h>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/string_cast.hpp>

// std
#include <iostream>
#include <cassert>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define STRATIFIED_X_SIZE 5
#define STRATIFIED_Y_SIZE 5
#define SPHERICAL_HARMONIC_BASIS_FUNCTIONS 9

namespace mcrt {

    Renderer::Renderer(Scene& scene, const Camera& camera): renderCamera{camera}, scene{scene}
    {
        initOptix();
        updateCamera(camera);

        std::cout << "Creating OptiX context..." << std::endl;
        createContext();

        std::cout << "Filling geometry buffers..." << std::endl;
        fillGeometryBuffers();

        std::cout << "Loading textures..." << std::endl;
        createTextures();

        std::cout << "Setting up pipeline..." << std::endl;
        GeometryBufferHandle geometryData = GeometryBufferHandle{ vertexBuffers, indexBuffers, normalBuffers, texcoordBuffers, textureObjects, amountVertices, amountIndices };

        std::vector<CUDABuffer> emptyNormalsVec;
        std::vector<CUDABuffer> emptyTexcoordBufferVec;
        std::vector<cudaTextureObject_t> emptyTexBufferVec;
        GeometryBufferHandle radianceCellGeometry = GeometryBufferHandle{ radianceGridVertexBuffers, radianceGridIndexBuffers, emptyNormalsVec, emptyTexcoordBufferVec, emptyTexBufferVec, amountVerticesRadianceGrid, amountIndicesRadianceGrid };

        tutorialPipeline = std::make_unique<DefaultPipeline>(optixContext, geometryData, scene);
        directLightPipeline = std::make_unique<DirectLightPipeline>(optixContext, geometryData, scene);
        radianceCellGatherPipeline = std::make_unique<RadianceCellGatherPipeline>(optixContext, radianceCellGeometry, geometryData, scene); // For now we can use normal geometry instead of proxy

        std::cout << "Context, module, pipeline, etc, all set up." << std::endl;
        std::cout << "MCRT renderer fully set up." << std::endl;

        // Direct lighting (preprocess)
        initDirectLightingTexture(1024);
        prepareUVWorldPositions();
        calculateDirectLighting();
        calculateRadianceCellGatherPass();
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

    void Renderer::createTextures()
    {
        int numTextures = (int)scene.getTextures().size();

        textureArrays.resize(numTextures);
        textureObjects.resize(numTextures);

        for (int textureID = 0; textureID < numTextures; textureID++) {
            auto texture = scene.getTextures()[textureID];

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
            textureObjects[textureID] = cuda_tex;
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
        // First resize needs to be done before rendering
        if (tutorialPipeline->launchParams.frame.size.x == 0) return;

        tutorialPipeline->uploadLaunchParams();

        // Launch render pipeline
        OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
            tutorialPipeline->pipeline, stream,
            /*! launch parameters and SBT */
            tutorialPipeline->launchParamsBuffer.d_pointer(),
            tutorialPipeline->launchParamsBuffer.sizeInBytes,
            &tutorialPipeline->sbt,
            /*! dimensions of the launch: */
            tutorialPipeline->launchParams.frame.size.x,
            tutorialPipeline->launchParams.frame.size.y,
            1
        ));


        // TODO: implement double buffering!!!
        // sync - make sure the frame is rendered before we download and
        // display (obviously, for a high-performance application you
        // want to use streams and double-buffering, but for this simple
        // example, this will have to do)
        CUDA_SYNC_CHECK();
    }

    void Renderer::updateCamera(const Camera& camera)
    {
        if (tutorialPipeline != nullptr)
        {
            renderCamera = camera;
            tutorialPipeline->launchParams.camera.position = camera.position;
            tutorialPipeline->launchParams.camera.direction = normalize(camera.target - camera.position);
            const float cosFovy = 0.66f;
            const float aspect = float(tutorialPipeline->launchParams.frame.size.x) / float(tutorialPipeline->launchParams.frame.size.y);
            tutorialPipeline->launchParams.camera.horizontal
                = cosFovy * aspect * normalize(cross(tutorialPipeline->launchParams.camera.direction,
                    camera.up));
            tutorialPipeline->launchParams.camera.vertical
                = cosFovy * normalize(cross(tutorialPipeline->launchParams.camera.horizontal,
                    tutorialPipeline->launchParams.camera.direction));
        }
    }

    void Renderer::writeToImage(std::string fileName, int resX, int resY, void* data)
    {
        stbi_write_png(fileName.c_str(), resX, resY, 4, data, resX * sizeof(uint32_t));
    }

    // Will allocate a `size * size` buffer on the GPU
    void Renderer::initDirectLightingTexture(int size)
    {
        std::vector<uint32_t> zeros(size * size, 0.0f);
        //directLightingTexture.resize(size * size * sizeof(uint32_t));
        directLightingTexture.alloc_and_upload(zeros);
        directLightPipeline->launchParams.directLightingTexture.size = size;
        directLightPipeline->launchParams.directLightingTexture.colorBuffer = (uint32_t*)directLightingTexture.d_pointer();
    }


    void Renderer::calculateDirectLighting()
    {
        // Get lights data from scene
        std::vector<LightData> lightData = scene.getLightsData();

        // Allocate device space for the light data buffer, then upload the light data to the device
        lightDataBuffer.resize(lightData.size() * sizeof(LightData));
        lightDataBuffer.upload(lightData.data(), 1);

        directLightPipeline->launchParams.amountLights = lightData.size();
        directLightPipeline->launchParams.lights = (LightData*)lightDataBuffer.d_pointer();
        directLightPipeline->launchParams.stratifyResX = STRATIFIED_X_SIZE;
        directLightPipeline->launchParams.stratifyResY = STRATIFIED_Y_SIZE;
        directLightPipeline->uploadLaunchParams();

        // Launch direct lighting pipeline
        OPTIX_CHECK(optixLaunch(
            directLightPipeline->pipeline, stream,
            directLightPipeline->launchParamsBuffer.d_pointer(),
            directLightPipeline->launchParamsBuffer.sizeInBytes,
            &directLightPipeline->sbt,
            directLightPipeline->launchParams.directLightingTexture.size,       // dimension X: x resolution of UV map
            directLightPipeline->launchParams.directLightingTexture.size,       // dimension Y: y resolution of UV map
            1                                                                   // dimension Z: 1
            // dimension X * dimension Y * dimension Z CUDA threads will be spawned 
        ));

        CUDA_SYNC_CHECK();

        // Download resulting texture from GPU
        std::vector<uint32_t> direct_lighting_result(directLightPipeline->launchParams.directLightingTexture.size * directLightPipeline->launchParams.directLightingTexture.size);
        directLightingTexture.download(direct_lighting_result.data(),
            directLightPipeline->launchParams.directLightingTexture.size * directLightPipeline->launchParams.directLightingTexture.size);


        // Flip Y coordinates, otherwise UV map is upside down
        std::vector<uint32_t> result_reversed;
    
        for (int y = directLightPipeline->launchParams.directLightingTexture.size - 1; y >= 0; y--)
        {
            for (int x = 0; x < directLightPipeline->launchParams.directLightingTexture.size; x++)
            {
                result_reversed.push_back(direct_lighting_result[y * directLightPipeline->launchParams.directLightingTexture.size + x]);
            }
        }
        

        //std::reverse(direct_lighting_result.begin(), direct_lighting_result.end());
        // Write the result to an image (for debugging purposes)
        writeToImage("direct_lighting_output.png", directLightPipeline->launchParams.directLightingTexture.size, directLightPipeline->launchParams.directLightingTexture.size, result_reversed.data());
    }

    void Renderer::initSHWeightsBuffer(int amountNonEmptyCells)
    {        
        // How to index: (nonEmptyCellIndex * 8 * SPHERICAL_HARMONIC_BASIS_FUNCTIONS) + (SHIndex * SPHERICAL_HARMONIC_BASIS_FUNCTIONS) + BasisFunctionIndex
        std::vector<double> shCoefficients(amountNonEmptyCells * 8 * SPHERICAL_HARMONIC_BASIS_FUNCTIONS, 0.0);
        SHWeightsDataBuffer.resize(shCoefficients.size() * sizeof(double));
        SHWeightsDataBuffer.alloc_and_upload(shCoefficients);
        radianceCellGatherPipeline->launchParams.sphericalHarmonicsWeights.weights = (double*)SHWeightsDataBuffer.d_pointer();
        radianceCellGatherPipeline->launchParams.sphericalHarmonicsWeights.size = amountNonEmptyCells * 8 * SPHERICAL_HARMONIC_BASIS_FUNCTIONS; // 8 SHs per cell, each 9 basis functions
        radianceCellGatherPipeline->launchParams.sphericalHarmonicsWeights.amountBasisFunctions = SPHERICAL_HARMONIC_BASIS_FUNCTIONS;

        std::cout << "Size of SH weights buffer on GPU in bytes: " << shCoefficients.size() * sizeof(double) << std::endl;
    }



    void Renderer::calculateRadianceCellGatherPass()
    {
        const int texSize = directLightPipeline->launchParams.directLightingTexture.size;

        // Initialize non-empty cell data on GPU
        NonEmptyCells nonEmpties = scene.grid.getNonEmptyCells();
        std::vector<glm::vec3> nonEmptyCellCenters;
        for (auto& nonEmpty : nonEmpties.nonEmptyCells)
        {
            nonEmptyCellCenters.push_back(nonEmpty->getCenter());
        }
        nonEmptyCellDataBuffer.resize(nonEmptyCellCenters.size() * sizeof(glm::vec3));
        nonEmptyCellDataBuffer.upload(nonEmptyCellCenters.data(), 1);
        radianceCellGatherPipeline->launchParams.nonEmptyCells.centers = (glm::vec3*)nonEmptyCellDataBuffer.d_pointer();
        radianceCellGatherPipeline->launchParams.nonEmptyCells.size = nonEmpties.nonEmptyCells.size();

        // Initialize Light Source Texture data on GPU
        // Take the direct lighting pass output as light source texture (for now), this should normally be the output from the previous pass in each iteration
        radianceCellGatherPipeline->launchParams.lightSourceTexture.colorBuffer = (uint32_t*)directLightingTexture.d_pointer();
        radianceCellGatherPipeline->launchParams.lightSourceTexture.size = texSize;

        // Initialize SH data on GPU
        initSHWeightsBuffer(nonEmptyCellCenters.size());

        // Initialize UV World positions data on GPU
        directLightPipeline->launchParams.uvWorldPositions.size = texSize * texSize;
        radianceCellGatherPipeline->launchParams.uvWorldPositions.UVDataBuffer = (UVWorldData*)UVWorldPositionDeviceBuffer.d_pointer();
        
        // Initialize cell size in launch params
        radianceCellGatherPipeline->launchParams.cellSize = scene.grid.getCellSize();

        // Initialize stratify cell sizes in launch params
        radianceCellGatherPipeline->launchParams.stratifyResX = STRATIFIED_X_SIZE;
        radianceCellGatherPipeline->launchParams.stratifyResY = STRATIFIED_Y_SIZE;

        radianceCellGatherPipeline->uploadLaunchParams();

        OPTIX_CHECK(optixLaunch(
            radianceCellGatherPipeline->pipeline, stream,
            radianceCellGatherPipeline->launchParamsBuffer.d_pointer(),
            radianceCellGatherPipeline->launchParamsBuffer.sizeInBytes,
            &radianceCellGatherPipeline->sbt,
            radianceCellGatherPipeline->launchParams.lightSourceTexture.size,   // dimension X: x resolution of light source UV map
            radianceCellGatherPipeline->launchParams.lightSourceTexture.size,   // dimension Y: y resolution of light source UV map
            1                                                                   // dimension Z: 1
            // dimension X * dimension Y * dimension Z CUDA threads will be spawned 
        ));

        CUDA_SYNC_CHECK();

        // Print SH results
        std::vector<double> shCoefficients(nonEmptyCellCenters.size() * 8 * SPHERICAL_HARMONIC_BASIS_FUNCTIONS);
        SHWeightsDataBuffer.download(shCoefficients.data(), nonEmptyCellCenters.size() * 8 * SPHERICAL_HARMONIC_BASIS_FUNCTIONS);

        for (int i = 0; i < nonEmptyCellCenters.size() * 8 * SPHERICAL_HARMONIC_BASIS_FUNCTIONS; i += 9)
        {
            std::cout << "[";
            for (int j = 0; j < 9; j++)
            {
                std::cout << shCoefficients[i + j] << ",";
            }
            std::cout << "]" << std::endl;
        }
    }   


    void Renderer::prepareUVWorldPositions()
    {
        const int texSize = directLightPipeline->launchParams.directLightingTexture.size;
        assert( (texSize > 0) && "Direct lighting texture needs to be initialized before preparing UV indices!");
        
        std::vector<UVWorldData> UVWorldPositions(texSize * texSize, { glm::vec3{-1000.0f, -1000.0f, -1000.0f}, glm::vec3{-1000.0f, -1000.0f, -1000.0f} });    // Scene is scaled within (0;1) so this should not form a problem

        for (int i = 0; i < UVWorldPositions.size(); i++)
        {
            const float u = float(i % texSize) / float(texSize);
            // (i - i % texSize) / texSize gives us the row number, divided by texSize gives us the V coordinate 
            const float v = (float((i - (i % texSize))) / float(texSize)) / float(texSize);
            glm::vec2 uv = glm::vec2{ u,v };
            UVWorldPositions[i] = UVto3D(uv);
        }

        // Upload world positions to the GPU and pass a pointer to this memory into the launch params
        UVWorldPositionDeviceBuffer.alloc_and_upload(UVWorldPositions);
        directLightPipeline->launchParams.uvWorldPositions.size = texSize * texSize;
        directLightPipeline->launchParams.uvWorldPositions.UVDataBuffer = (UVWorldData*)UVWorldPositionDeviceBuffer.d_pointer();
    }

    float Renderer::area(glm::vec2 a, glm::vec2 b, glm::vec2 c)
    {
        // (w * h) / 2
        glm::vec2 v1 = a - c;
        glm::vec2 v2 = b - c;
        return (v1.x * v2.y - v1.y * v2.x) / 2.0f;
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

                float a = area(uv1, uv2, uv3);
                if (a == 0.0f) continue;

                // Barycentric coordinates
                float a1 = area(uv2, uv3, uv) / a; if (a1 < 0) continue;
                float a2 = area(uv3, uv1, uv) / a; if (a2 < 0) continue;
                float a3 = area(uv1, uv2, uv) / a; if (a3 < 0) continue;

                //std::cout << "UV found!" << std::endl;
                glm::vec3 uvPosition = a1 * g->model->mesh->vertices[triangle[0]] + a2 * g->model->mesh->vertices[triangle[1]] + a3 * g->model->mesh->vertices[triangle[2]];
                glm::vec3 uvNormal = glm::normalize(a1 * g->model->mesh->normals[triangle[0]] + a2 * g->model->mesh->normals[triangle[1]] + a3 * g->model->mesh->normals[triangle[2]]);
                uvPosition = g->worldTransform.object2World * glm::vec4{ uvPosition, 1.0f };
                return { uvPosition, uvNormal};
            }
        }
        return { glm::vec3{ -1000.0f, -1000.0f, -1000.0f }, glm::vec3{ -1000.0f, -1000.0f, -1000.0f } };
    }



    void Renderer::resize(const glm::ivec2& newSize)
    {
        // If window minimized
        if (newSize.x == 0 | newSize.y == 0) return;

        // Resize CUDA frame buffer
        colorBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));
    
        // Update launch parameters that are passed to OptiX launch
        tutorialPipeline->launchParams.frame.size = newSize;
        tutorialPipeline->launchParams.frame.colorBuffer = (uint32_t*)colorBuffer.d_pointer();

        // Reset camera, aspect may have changed
        updateCamera(renderCamera);
    }

    // Copy rendered color buffer from device to host memory for display
    void Renderer::downloadPixels(uint32_t h_pixels[])
    {
        colorBuffer.download(h_pixels,
            tutorialPipeline->launchParams.frame.size.x * tutorialPipeline->launchParams.frame.size.y);
    }

    void Renderer::downloadDirectLighting(uint32_t h_pixels[])
    {
        directLightingTexture.download(h_pixels,
            directLightPipeline->launchParams.directLightingTexture.size * directLightPipeline->launchParams.directLightingTexture.size);
    }

}