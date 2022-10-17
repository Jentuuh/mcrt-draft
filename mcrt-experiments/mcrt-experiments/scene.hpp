#pragma once
#include <vector>
#include "glm/glm.hpp"
#include "game_object.hpp"
namespace mcrt {
	class Scene
	{
	public:
		Scene();

		//std::vector<glm::vec3>& vertices() { return sceneVertices; };
		//std::vector<glm::ivec3>& indices() { return sceneIndices; };
		glm::vec3 maxCoord() { return sceneMax; };
		glm::vec3 minCoord() { return sceneMin; };
		int numObjects() { return gameObjects.size(); };
		int amountVertices();
		std::vector<GameObject>& getGameObjects() { return gameObjects; };

		void addGameObject(glm::vec3 position, glm::vec3 rotation, glm::vec3 scale, std::shared_ptr<Model> model);
		void normalize();


	private:
		std::vector<GameObject> gameObjects;

		//std::vector<glm::vec3> sceneVertices;
		//std::vector<glm::ivec3> sceneIndices;

		glm::vec3 sceneMax;
		glm::vec3 sceneMin;
	};

}

