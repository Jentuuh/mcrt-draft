#include "scene.hpp"
#include "glm/gtx/string_cast.hpp"
#include "helpers.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

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
			amountVertices += g.amountVertices();
		}
		return amountVertices;
	}

	void Scene::addGameObject(glm::vec3 position, glm::vec3 rotation, glm::vec3 scale, std::shared_ptr<Model> model)
	{

		GameObject newObj = GameObject{ Transform{position, rotation, scale}, model };
		gameObjects.push_back(newObj);

		// Record scene bounds (for normalization afterwards)
		std::vector<glm::vec3> newVertices = newObj.getWorldVertices();
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

	void Scene::buildRadianceGrid(float cellSize)
	{
		grid.init(cellSize);
	}

	// Make sure this method is called AFTER all objects are added to the scene
	void Scene::normalize()
	{
		std::cout << "Normalizing scene..." << std::endl;

		float scaleX = 1.0f/abs(sceneMax.x - sceneMin.x);
		float scaleY = 1.0f/abs(sceneMax.y - sceneMin.y);
		float scaleZ = 1.0f/abs(sceneMax.z - sceneMin.z);

		// We need to assure uniform scaling (otherwise objects will be deformed, 
		// so we take the largest downscale factor as our uniform scaling factor.
		float minScale = std::min(scaleX, scaleY);
		minScale = std::min(minScale, scaleZ);


		glm::vec3 translation = glm::vec3{ 0.0f, 0.0f, 0.0f } - sceneMin;

		std::cout << "Shifting scene by: " << glm::to_string(translation) << std::endl;

		std::cout << "Uniformly downscaling by: " << minScale << std::endl;

		for (auto& g : gameObjects)
		{
			g.worldTransform.translate(translation);
			g.worldTransform.applySceneRescale(glm::vec3{ minScale, minScale, minScale });
		}

	}


    /*! find vertex with given position, normal, texcoord, and return
    its vertex ID, or, if it doesn't exit, add it to the mesh, and
    its just-created index */
    int addVertex(std::shared_ptr<TriangleMesh> mesh,
        tinyobj::attrib_t& attributes,
        const tinyobj::index_t& idx,
        std::map<tinyobj::index_t, int>& knownVertices)
    {
        if (knownVertices.find(idx) != knownVertices.end())
            return knownVertices[idx];

        const glm::vec3* vertex_array = (const glm::vec3*)attributes.vertices.data();
        const glm::vec3* normal_array = (const glm::vec3*)attributes.normals.data();
        const glm::vec2* texcoord_array = (const glm::vec2*)attributes.texcoords.data();

        int newID = mesh->vertices.size();
        knownVertices[idx] = newID;

        mesh->vertices.push_back(vertex_array[idx.vertex_index]);
        if (idx.normal_index >= 0) {
            while (mesh->normals.size() < mesh->vertices.size())
                mesh->normals.push_back(normal_array[idx.normal_index]);
        }
        if (idx.texcoord_index >= 0) {
            while (mesh->texCoords.size() < mesh->vertices.size())
                mesh->texCoords.push_back(texcoord_array[idx.texcoord_index]);
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
        const std::string mtlDir
            = fileName.substr(0, fileName.rfind('/') + 1);

        std::cout << "Loading OBJ file from: " << mtlDir << std::endl;

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
                mtlDir.c_str(),
                /* triangulate */true);
        if (!readOK) {
            throw std::runtime_error("Could not read OBJ model from " + fileName + ":" + mtlDir + " : " + err);
        }

        if (materials.empty())
            throw std::runtime_error("could not parse materials ...");

        std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;
        for (int shapeID = 0; shapeID < (int)shapes.size(); shapeID++) {
            tinyobj::shape_t& shape = shapes[shapeID];

            std::set<int> materialIDs;
            for (auto faceMatID : shape.mesh.material_ids)
                materialIDs.insert(faceMatID);

            std::map<tinyobj::index_t, int> knownVertices;

            for (int materialID : materialIDs) {
                // We split each different material up in a different gameObject
                std::shared_ptr<TriangleMesh> mesh = std::make_shared<TriangleMesh>();

                for (int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++) {
                    if (shape.mesh.material_ids[faceID] != materialID) continue;
                    tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
                    tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
                    tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

                    glm::ivec3 idx(addVertex(mesh, attributes, idx0, knownVertices),
                        addVertex(mesh, attributes, idx1, knownVertices),
                        addVertex(mesh, attributes, idx2, knownVertices));
                    mesh->indices.push_back(idx);
                    mesh->diffuse = (const glm::vec3&)materials[materialID].diffuse;
                    mesh->diffuse = generateRandomColor();
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


}