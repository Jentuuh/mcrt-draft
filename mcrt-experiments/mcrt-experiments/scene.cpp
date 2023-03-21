#pragma once
#include "scene.hpp"
#include "glm/gtx/string_cast.hpp"
#include "helpers.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <stb/stb_image.h>

#include <set>
#include <limits>
#include <iostream>

namespace std {
    inline bool operator<(const tinyobj::index_t& a,
        const tinyobj::index_t& b)
    {
        if (a.vertex_index < b.vertex_index) return true;
        if (a.vertex_index > b.vertex_index) return false;

        if (a.normal_index < b.normal_index) return true;
        if (a.normal_index > b.normal_index) return false;

        if (a.texcoord_index < b.texcoord_index) return true;
        if (a.texcoord_index > b.texcoord_index) return false;

        return false;
    }
}

namespace mcrt {

	Scene::Scene()
	{
		grid = RadianceGrid{};	// Note that the radiance grid still needs to be built after the scene geometry is completed!
		sceneMin = glm::vec3{ std::numeric_limits<float>::infinity() };
		sceneMax = glm::vec3{ -std::numeric_limits<float>::infinity() };
	}


	int Scene::amountVertices()
	{
		int amountVertices = 0;

		for (auto& g : gameObjects)
		{
			amountVertices += g->amountVertices();
		}
		return amountVertices;
	}

    std::vector<LightData> Scene::getLightsData()
    {
        std::vector<LightData> lightsData;
        for (auto& l : lights)
        {
            lightsData.push_back(l.lightProps);
        }
        return lightsData;
    }


    void Scene::loadSponzaComponents()
    {
     /*   loadModelFromOBJ("../models/sponza-components/arcs.obj");
        loadModelFromOBJ("../models/sponza-components/arcs2.obj");
        loadModelFromOBJ("../models/sponza-components/building_core.obj");
        loadModelFromOBJ("../models/sponza-components/bushes.obj");
        loadModelFromOBJ("../models/sponza-components/curtains.obj");
        loadModelFromOBJ("../models/sponza-components/curtains2.obj");
        loadModelFromOBJ("../models/sponza-components/curtains3.obj");
        loadModelFromOBJ("../models/sponza-components/doors.obj");
        loadModelFromOBJ("../models/sponza-components/drapes.obj");
        loadModelFromOBJ("../models/sponza-components/drapes2.obj");
        loadModelFromOBJ("../models/sponza-components/drapes3.obj");
        loadModelFromOBJ("../models/sponza-components/fire_pit.obj");
        loadModelFromOBJ("../models/sponza-components/lion.obj");
        loadModelFromOBJ("../models/sponza-components/lion_background.obj");
        loadModelFromOBJ("../models/sponza-components/pillars.obj");
        loadModelFromOBJ("../models/sponza-components/pillars_up.obj");
        loadModelFromOBJ("../models/sponza-components/pillars_up2.obj");
        loadModelFromOBJ("../models/sponza-components/plants.obj");
        loadModelFromOBJ("../models/sponza-components/platforms.obj");
        loadModelFromOBJ("../models/sponza-components/pots.obj");
        loadModelFromOBJ("../models/sponza-components/roof.obj");
        loadModelFromOBJ("../models/sponza-components/spears.obj");
        loadModelFromOBJ("../models/sponza-components/spikes.obj");
        loadModelFromOBJ("../models/sponza-components/square_panel_back.obj");
        loadModelFromOBJ("../models/sponza-components/water_dish.obj");*/

        loadModelFromOBJWithMultipleUVs("../models/sponza-components/arcs.obj", "../models/sponza-components-textured/arcs.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/arcs2.obj", "../models/sponza-components-textured/arcs2.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/building_core.obj", "../models/sponza-components-textured/building_core.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/bushes.obj", "../models/sponza-components-textured/bushes.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/curtains.obj", "../models/sponza-components-textured/curtains.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/curtains2.obj", "../models/sponza-components-textured/curtains2.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/curtains3.obj", "../models/sponza-components-textured/curtains3.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/doors.obj", "../models/sponza-components-textured/doors.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/drapes.obj", "../models/sponza-components-textured/drapes.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/drapes2.obj","../models/sponza-components-textured/drapes2.obj" );
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/drapes3.obj", "../models/sponza-components-textured/drapes3.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/fire_pit.obj", "../models/sponza-components-textured/fire_pit.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/lion.obj", "../models/sponza-components-textured/lion.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/lion_background.obj", "../models/sponza-components-textured/lion_background.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/pillars.obj", "../models/sponza-components-textured/pillars.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/pillars_up.obj", "../models/sponza-components-textured/pillars_up.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/pillars_up2.obj", "../models/sponza-components-textured/pillars_up2.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/plants.obj", "../models/sponza-components-textured/plants.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/platforms.obj", "../models/sponza-components-textured/platforms.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/pots.obj", "../models/sponza-components-textured/pots.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/roof.obj", "../models/sponza-components-textured/roof.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/spears.obj", "../models/sponza-components/spears.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/spikes.obj", "../models/sponza-components/spikes.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/square_panel_back.obj", "../models/sponza-components/square_panel_back.obj");
        loadModelFromOBJWithMultipleUVs("../models/sponza-components/water_dish.obj", "../models/sponza-components/water_dish.obj");

    }



	void Scene::addGameObject(glm::vec3 position, glm::vec3 rotation, glm::vec3 scale, std::shared_ptr<Model> model)
	{

		std::shared_ptr<GameObject> newObj = std::make_shared<GameObject>( Transform{position, rotation, scale}, model);
		gameObjects.push_back(newObj);

		// Record scene bounds (for normalization afterwards)
		std::vector<glm::vec3> newVertices = newObj->getWorldVertices();
		for (auto& v : newVertices)
		{
			if (v.x > sceneMax.x) sceneMax.x = v.x;
			if (v.y > sceneMax.y) sceneMax.y = v.y;
			if (v.z > sceneMax.z) sceneMax.z = v.z;

			if (v.x < sceneMin.x) sceneMin.x = v.x;
			if (v.y < sceneMin.y) sceneMin.y = v.y;
			if (v.z < sceneMin.z) sceneMin.z = v.z;
		}
	}

    void Scene::voxelizeObjects()
    {
        for (auto g : gameObjects)
        {
            std::shared_ptr<Voxelizer> voxelizer = std::make_shared<Voxelizer>( 0.01f, g );
            voxelizer->voxelize();
            voxelizers.push_back(voxelizer);
        }
    }


	void Scene::buildRadianceGrid(float cellSize)
	{
		grid.init(cellSize);
	}

	// Make sure this method is called AFTER all objects are added to the scene
	void Scene::normalize()
	{
		std::cout << "Normalizing scene..." << std::endl;

		float scaleX = 0.99f / abs(sceneMax.x - sceneMin.x);
		float scaleY = 0.99f / abs(sceneMax.y - sceneMin.y);
		float scaleZ = 0.99f / abs(sceneMax.z - sceneMin.z);

        //float scaleX = 1.0f / abs(sceneMax.x - sceneMin.x);
        //float scaleY = 1.0f / abs(sceneMax.y - sceneMin.y);
        //float scaleZ = 1.0f / abs(sceneMax.z - sceneMin.z);

		// We need to assure uniform scaling (otherwise objects will be deformed, 
		// so we take the largest downscale factor as our uniform scaling factor.
		float minScale = std::min(scaleX, scaleY);
		minScale = std::min(minScale, scaleZ);


		glm::vec3 translation = glm::vec3{ 0.0001f, 0.0001f, 0.0001f } - sceneMin;

		std::cout << "Shifting scene by: " << glm::to_string(translation) << std::endl;

		std::cout << "Uniformly downscaling by: " << minScale << std::endl;

		for (auto& g : gameObjects)
		{
			g->translate(translation);
			g->worldTransform.applySceneRescale(glm::vec3{ minScale, minScale, minScale });
		}

	}

    void Scene::loadLights()
    {
        // LightData{origin, du, dv, normal, power, width, height}
        //lights.push_back(AreaLight{ false, LightData{{0.4f, 0.4f, 0.88f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, {0.8f, 0.8f, 0.8f}, 0.2f, 0.2f} });
        lights.push_back(AreaLight{ false, LightData{{0.45f, 0.965f, 0.45f }, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, -1.0f, 0.0f}, {2.0f, 2.0f, 2.0f}, 0.1f, 0.1f} });
    }


    /*! find vertex with given position, normal, texcoord, and return
    its vertex ID, or, if it doesn't exit, add it to the mesh, and
    its just-created index */
    int addVertexTextured(std::shared_ptr<TriangleMesh> mesh,
        tinyobj::attrib_t& attributes,
        const tinyobj::index_t& idx,
        std::map<tinyobj::index_t, int>& knownVertices,
        std::map<int, glm::vec2>& diffuseUVVertexIndexMap)
    {
        if (knownVertices.find(idx) != knownVertices.end())
            return knownVertices[idx];

        const glm::vec3* vertex_array = (const glm::vec3*)attributes.vertices.data();
        const glm::vec3* normal_array = (const glm::vec3*)attributes.normals.data();
        const glm::vec2* texcoord_array = (const glm::vec2*)attributes.texcoords.data();

        int newID = mesh->vertices.size();
        knownVertices[idx] = newID;

        // Update mesh's bounding box
        if (vertex_array[idx.vertex_index].x < mesh->boundingBox.min.x)
        {
            mesh->boundingBox.min.x = vertex_array[idx.vertex_index].x;
        }
        if (vertex_array[idx.vertex_index].x > mesh->boundingBox.max.x)
        {
            mesh->boundingBox.max.x = vertex_array[idx.vertex_index].x;
        }
        if (vertex_array[idx.vertex_index].y < mesh->boundingBox.min.y)
        {
            mesh->boundingBox.min.y = vertex_array[idx.vertex_index].y;
        }
        if (vertex_array[idx.vertex_index].y > mesh->boundingBox.max.y)
        {
            mesh->boundingBox.max.y = vertex_array[idx.vertex_index].y;
        }
        if (vertex_array[idx.vertex_index].z < mesh->boundingBox.min.z)
        {
            mesh->boundingBox.min.z = vertex_array[idx.vertex_index].z;
        }
        if (vertex_array[idx.vertex_index].z > mesh->boundingBox.max.z)
        {
            mesh->boundingBox.max.z = vertex_array[idx.vertex_index].z;
        }

        mesh->vertices.push_back(vertex_array[idx.vertex_index]);
        if (idx.normal_index >= 0) {
            while (mesh->normals.size() < mesh->vertices.size())
                mesh->normals.push_back(normal_array[idx.normal_index]);
        }
        if (idx.texcoord_index >= 0) {
            while (mesh->texCoords.size() < mesh->vertices.size())
            {
                mesh->texCoords.push_back(texcoord_array[idx.texcoord_index]);
                // Set a value for the current vertex index in the map, so we know which diffuse UV belongs to that vertex.
                // We can then later use this information to fill the diffuseUV vector for the lightMapped mesh correctly.
                diffuseUVVertexIndexMap[idx.vertex_index] = texcoord_array[idx.texcoord_index];     
            }
        }

        // just for sanity's sake:
        if (mesh->texCoords.size() > 0)
            mesh->texCoords.resize(mesh->vertices.size());
        // just for sanity's sake:
        if (mesh->normals.size() > 0)
            mesh->normals.resize(mesh->vertices.size());

        return newID;
    }

    /*! find vertex with given position, normal, texcoord, and return
   its vertex ID, or, if it doesn't exit, add it to the mesh, and
   its just-created index */
    int addVertexLightMapped(std::shared_ptr<TriangleMesh> mesh,
        tinyobj::attrib_t& attributes,
        const tinyobj::index_t& idx,
        std::map<tinyobj::index_t, int>& knownVertices,
        std::map<int, glm::vec2>& diffuseUVVertexIndexMap)
    {
        if (knownVertices.find(idx) != knownVertices.end())
            return knownVertices[idx];

        const glm::vec3* vertex_array = (const glm::vec3*)attributes.vertices.data();
        const glm::vec3* normal_array = (const glm::vec3*)attributes.normals.data();
        const glm::vec2* texcoord_array = (const glm::vec2*)attributes.texcoords.data();

        int newID = mesh->vertices.size();
        knownVertices[idx] = newID;

        // Update mesh's bounding box
        if (vertex_array[idx.vertex_index].x < mesh->boundingBox.min.x)
        {
            mesh->boundingBox.min.x = vertex_array[idx.vertex_index].x;
        }
        if (vertex_array[idx.vertex_index].x > mesh->boundingBox.max.x)
        {
            mesh->boundingBox.max.x = vertex_array[idx.vertex_index].x;
        }
        if (vertex_array[idx.vertex_index].y < mesh->boundingBox.min.y)
        {
            mesh->boundingBox.min.y = vertex_array[idx.vertex_index].y;
        }
        if (vertex_array[idx.vertex_index].y > mesh->boundingBox.max.y)
        {
            mesh->boundingBox.max.y = vertex_array[idx.vertex_index].y;
        }
        if (vertex_array[idx.vertex_index].z < mesh->boundingBox.min.z)
        {
            mesh->boundingBox.min.z = vertex_array[idx.vertex_index].z;
        }
        if (vertex_array[idx.vertex_index].z > mesh->boundingBox.max.z)
        {
            mesh->boundingBox.max.z = vertex_array[idx.vertex_index].z;
        }

        mesh->vertices.push_back(vertex_array[idx.vertex_index]);
        if (idx.normal_index >= 0) {
            while (mesh->normals.size() < mesh->vertices.size())
                mesh->normals.push_back(normal_array[idx.normal_index]);
        }
        if (idx.texcoord_index >= 0) {
            while (mesh->texCoords.size() < mesh->vertices.size())
                mesh->texCoords.push_back(texcoord_array[idx.texcoord_index]);
        }

        // If this vertex has a diffuse UV coordinate in the textured mesh
        if (diffuseUVVertexIndexMap.find(idx.vertex_index) != diffuseUVVertexIndexMap.end())
        {
            // For each vertex that we add, we need to give it a diffuse UV coordinate, namely, the one that 
            // corresponds to the same vertex (vertex index) from the textured mesh.
            mesh->diffuseTextureCoords.push_back(diffuseUVVertexIndexMap[idx.vertex_index]);
        }
        else {
            mesh->diffuseTextureCoords.push_back(glm::vec2{ -1.0f, -1.0f });
        }

        // just for sanity's sake:
        if (mesh->texCoords.size() > 0)
            mesh->texCoords.resize(mesh->vertices.size());
        // just for sanity's sake:
        if (mesh->normals.size() > 0)
            mesh->normals.resize(mesh->vertices.size());

        return newID;
    }



    void Scene::loadModelFromOBJ(const std::string& fileName)
    {
        const std::string modelDir
            = fileName.substr(0, fileName.rfind('/') + 1);

        std::cout << "Loading OBJ file from: " << modelDir << std::endl;

        tinyobj::attrib_t attributes;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string err = "";

        bool readOK
            = tinyobj::LoadObj(&attributes,
                &shapes,
                &materials,
                &err,
                &err,
                fileName.c_str(),
                modelDir.c_str(),
                /* triangulate */true);
        if (!readOK) {
            throw std::runtime_error("Could not read OBJ model from " + fileName + ":" + modelDir + " : " + err);
        }

        if (materials.empty())
            throw std::runtime_error("could not parse materials ...");

        std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;
        std::map<std::string, int> knownTextures;

        for (int shapeID = 0; shapeID < (int)shapes.size(); shapeID++) {
            tinyobj::shape_t& shape = shapes[shapeID];

            std::set<int> materialIDs;
            for (auto faceMatID : shape.mesh.material_ids)
                materialIDs.insert(faceMatID);

            std::map<tinyobj::index_t, int> knownVertices;

            for (int materialID : materialIDs) {
                // We split each different material up in a different gameObject
                std::shared_ptr<TriangleMesh> mesh = std::make_shared<TriangleMesh>();
                std::map<int, glm::vec2> diffuseUVVertexIndexMap;

                for (int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++) {
                    if (shape.mesh.material_ids[faceID] != materialID) continue;
                    tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
                    tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
                    tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

                    glm::ivec3 idx(addVertexTextured(mesh, attributes, idx0, knownVertices, diffuseUVVertexIndexMap),
                        addVertexTextured(mesh, attributes, idx1, knownVertices, diffuseUVVertexIndexMap),
                        addVertexTextured(mesh, attributes, idx2, knownVertices, diffuseUVVertexIndexMap));
                    mesh->indices.push_back(idx);
                    mesh->diffuse = (const glm::vec3&)materials[materialID].diffuse;
                    //mesh->diffuse = generateRandomColor();
                    mesh->diffuseTextureID = loadTexture(knownTextures,
                        materials[materialID].diffuse_texname,
                        modelDir);
                }

                if (mesh->vertices.empty())
                {
                    mesh.reset();
                }
                else {
                    std::shared_ptr<Model> model = std::make_shared<Model>();
                    model->mesh = mesh;
                    addGameObject(glm::vec3{ 0.0f, 0.0f, 0.0f }, glm::vec3{ 0.0f, 0.0f, 0.0f }, glm::vec3{ 1.0f, 1.0f, 1.0f }, model);
                }
            }
        }

        std::cout << "Succesfully loaded in scene model. Resulted in " << gameObjects.size() << " game objects." << std::endl;
    }

    // Because .OBJ file does not support multiple UV layers, we load one version of the model with lightmap UVs and one version with texture UVs
    // and combine the data into our engine's GameObject
    void Scene::loadModelFromOBJWithMultipleUVs(const std::string& fileNameLightMappedUVs, const std::string& fileNameTextureUVs)
    {
        int textureID;
        std::map<int, glm::vec2> diffuseUVVertexIndexMap;

        // =============================================
        //            LOAD TEXTURED OBJECT
        // =============================================
        const std::string modelDirTextured
            = fileNameTextureUVs.substr(0, fileNameTextureUVs.rfind('/') + 1);

        std::cout << "Loading OBJ file from: " << fileNameTextureUVs << std::endl;

        tinyobj::attrib_t attributesTextured;
        std::vector<tinyobj::shape_t> shapesTextured;
        std::vector<tinyobj::material_t> materialsTextured;
        std::string err = "";

        bool readOK
            = tinyobj::LoadObj(&attributesTextured,
                &shapesTextured,
                &materialsTextured,
                &err,
                &err,
                fileNameTextureUVs.c_str(),
                modelDirTextured.c_str(),
                /* triangulate */true);
        if (!readOK) {
            throw std::runtime_error("Could not read OBJ model from " + fileNameTextureUVs + ":" + modelDirTextured + " : " + err);
        }

        if (materialsTextured.empty())
            throw std::runtime_error("could not parse materials ...");

        std::map<std::string, int> knownTextures;

        for (int shapeID = 0; shapeID < (int)shapesTextured.size(); shapeID++) {
            tinyobj::shape_t& shape = shapesTextured[shapeID];

            std::set<int> materialIDs;
            for (auto faceMatID : shape.mesh.material_ids)
                materialIDs.insert(faceMatID);

            std::map<tinyobj::index_t, int> knownVertices;

            for (int materialID : materialIDs) {
                // We split each different material up in a different gameObject
                std::shared_ptr<TriangleMesh> meshTextured = std::make_shared<TriangleMesh>();

                for (int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++) {
                    if (shape.mesh.material_ids[faceID] != materialID) continue;
                    tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
                    tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
                    tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

                    glm::ivec3 idx(addVertexTextured(meshTextured, attributesTextured, idx0, knownVertices, diffuseUVVertexIndexMap),
                        addVertexTextured(meshTextured, attributesTextured, idx1, knownVertices, diffuseUVVertexIndexMap),
                        addVertexTextured(meshTextured, attributesTextured, idx2, knownVertices, diffuseUVVertexIndexMap));
                    meshTextured->indices.push_back(idx);
                    meshTextured->diffuse = (const glm::vec3&)materialsTextured[materialID].diffuse;
                    //mesh->diffuse = generateRandomColor();
                    meshTextured->diffuseTextureID = loadTexture(knownTextures,
                        materialsTextured[materialID].diffuse_texname,
                        modelDirTextured);
                }

                if (meshTextured->vertices.empty())
                {
                    meshTextured.reset();
                }
                else {
                    // We do not add the textured version of the model to the scene, we just wanted to extract the UVs and load in the textures
                    textureID = meshTextured->diffuseTextureID;
                }
            }
        }

        // ===========================================================================
        //            LOAD LIGHTMAPPED OBJECT (we'll actually use this as geometry)
        // ===========================================================================
        const std::string modelDirLightMapped
            = fileNameLightMappedUVs.substr(0, fileNameLightMappedUVs.rfind('/') + 1);

        std::cout << "Loading OBJ file from: " << fileNameLightMappedUVs << std::endl;

        tinyobj::attrib_t attributesLightMapped;
        std::vector<tinyobj::shape_t> shapesLightMapped;
        std::vector<tinyobj::material_t> materialsLightMapped;
        err = "";

        readOK
            = tinyobj::LoadObj(&attributesLightMapped,
                &shapesLightMapped,
                &materialsLightMapped,
                &err,
                &err,
                fileNameLightMappedUVs.c_str(),
                modelDirLightMapped.c_str(),
                /* triangulate */true);
        if (!readOK) {
            throw std::runtime_error("Could not read OBJ model from " + fileNameLightMappedUVs + ":" + modelDirLightMapped + " : " + err);
        }

        if (materialsLightMapped.empty())
            throw std::runtime_error("could not parse materials ...");

        std::cout << "Done loading obj file - found " << shapesLightMapped.size() << " shapes with " << materialsLightMapped.size() << " materials" << std::endl;

        for (int shapeID = 0; shapeID < (int)shapesLightMapped.size(); shapeID++) {
            tinyobj::shape_t& shape = shapesLightMapped[shapeID];

            std::set<int> materialIDs;
            for (auto faceMatID : shape.mesh.material_ids)
                materialIDs.insert(faceMatID);

            std::map<tinyobj::index_t, int> knownVertices;

            for (int materialID : materialIDs) {
                // We split each different material up in a different gameObject
                std::shared_ptr<TriangleMesh> meshLightMapped = std::make_shared<TriangleMesh>();

                for (int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++) {
                    if (shape.mesh.material_ids[faceID] != materialID) continue;
                    tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
                    tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
                    tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

                    glm::ivec3 idx(addVertexLightMapped(meshLightMapped, attributesLightMapped, idx0, knownVertices, diffuseUVVertexIndexMap),
                        addVertexLightMapped(meshLightMapped, attributesLightMapped, idx1, knownVertices, diffuseUVVertexIndexMap),
                        addVertexLightMapped(meshLightMapped, attributesLightMapped, idx2, knownVertices, diffuseUVVertexIndexMap));
                    meshLightMapped->indices.push_back(idx);
                    meshLightMapped->diffuse = (const glm::vec3&)materialsLightMapped[materialID].diffuse;
                    meshLightMapped->diffuseTextureID = textureID;
                }

                if (meshLightMapped->vertices.empty())
                {
                    meshLightMapped.reset();
                }
                else {
                    std::shared_ptr<Model> model = std::make_shared<Model>();
                    model->mesh = meshLightMapped;
                    addGameObject(glm::vec3{ 0.0f, 0.0f, 0.0f }, glm::vec3{ 0.0f, 0.0f, 0.0f }, glm::vec3{ 1.0f, 1.0f, 1.0f }, model);
                }
            }
        }
        std::cout << "Succesfully loaded in scene model. Resulted in " << gameObjects.size() << " game objects." << std::endl;
    }


    int Scene::loadTexture(std::map<std::string, int>&knownTextures, const std::string & inFileName, const std::string & modelPath)
    {
        if (inFileName == "")
            return -1;

        if (knownTextures.find(inFileName) != knownTextures.end())
            return knownTextures[inFileName];

        std::string fileName = inFileName;
        // first, fix backspaces:
        for (auto& c : fileName)
            if (c == '\\') c = '/';

        // Deal with leading trail character
        if (modelPath[modelPath.length() - 1] == '/')
        {
            fileName = modelPath + fileName;
        } else{
            fileName = modelPath + "/" + fileName;
        }

        glm::ivec2 res;
        int   comp;
        unsigned char* image = stbi_load(fileName.c_str(),
            &res.x, &res.y, &comp, STBI_rgb_alpha);
        int textureID = -1;
        if (image) {
            textureID = (int)textures.size();
            std::shared_ptr<Texture> texture = std::make_shared<Texture>();
            texture->resolution = res;
            texture->pixel = (uint32_t*)image;

            /* iw - actually, it seems that stbi loads the pictures
               mirrored along the y axis - mirror them here */
            for (int y = 0; y < res.y / 2; y++) {
                uint32_t* line_y = texture->pixel + y * res.x;
                uint32_t* mirrored_y = texture->pixel + (res.y - 1 - y) * res.x;
                int mirror_y = res.y - 1 - y;
                for (int x = 0; x < res.x; x++) {
                    std::swap(line_y[x], mirrored_y[x]);
                }
            }

            textures.push_back(texture);
        }
        else {
            std::cout << "Could not load texture from " << fileName << "!" << std::endl;
        }

        knownTextures[inFileName] = textureID;
        return textureID;
    }



}