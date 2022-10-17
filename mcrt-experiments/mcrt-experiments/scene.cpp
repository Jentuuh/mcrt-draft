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