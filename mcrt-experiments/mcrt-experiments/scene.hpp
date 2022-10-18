#pragma once
#include <vector>
#include "glm/glm.hpp"
#include "game_object.hpp"
#include "radiance_grid.hpp"

namespace mcrt {
	class Scene
	{
	public:
		Scene();

		glm::vec3 maxCoord() { return sceneMax; };
		glm::vec3 minCoord() { return sceneMin; };
		int numObjects() { return gameObjects.size(); };
		int amountVertices();
		std::vector<GameObject>& getGameObjects() { return gameObjects; };

		void addGameObject(glm::vec3 position, glm::vec3 rotation, glm::vec3 scale, std::shared_ptr<Model> model);
		void loadModelFromOBJ(const std::string& fileName);
		void buildRadianceGrid(float cellSize);
		void normalize();

		RadianceGrid grid;
	private:
		std::vector<GameObject> gameObjects;

		glm::vec3 sceneMax;
		glm::vec3 sceneMin;
	};

}

