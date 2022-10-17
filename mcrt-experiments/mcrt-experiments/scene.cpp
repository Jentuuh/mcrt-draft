#include "scene.hpp"
#include "glm/gtx/string_cast.hpp"

#include <limits>
#include <iostream>

namespace mcrt {

	Scene::Scene()
	{
		sceneMin = glm::vec3{ std::numeric_limits<float>::infinity() };
		sceneMax = glm::vec3{ -std::numeric_limits<float>::infinity() };
	}

	void Scene::addGameObject(glm::vec3 position, glm::vec3 rotation, glm::vec3 scale, std::shared_ptr<Model> model)
	{

		GameObject newObj = GameObject{ Transform{position, rotation, scale}, model };
		gameObjects.push_back(newObj);

		std::vector<glm::vec3> newVertices = newObj.getWorldVertices();

		// Push back vertices (This will only work for a cube at the moment)
		//int indexOffset = sceneVertices.size();
		for (auto& v : newVertices)
		{
			if (v.x > sceneMax.x) sceneMax.x = v.x;
			if (v.y > sceneMax.y) sceneMax.y = v.y;
			if (v.z > sceneMax.z) sceneMax.z = v.z;

			if (v.x < sceneMin.x) sceneMin.x = v.x;
			if (v.y < sceneMin.y) sceneMin.y = v.y;
			if (v.z < sceneMin.z) sceneMin.z = v.z;
			//sceneVertices.push_back(v);
		}

		// Push back indices (This will only work for a cube at the moment)
		int indicesCube[] = { 0,1,3, 2,3,0,
							 5,7,6, 5,6,4,
							 0,4,5, 0,5,1,
							 2,3,7, 2,7,6,
							 1,5,7, 1,7,3,
							 4,0,2, 4,2,6
							};

		//for (int i = 0; i < 12; i++)
		//	sceneIndices.push_back(indexOffset + glm::ivec3(indicesCube[3 * i + 0],
		//												indicesCube[3 * i + 1],
		//												indicesCube[3 * i + 2]));
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


}