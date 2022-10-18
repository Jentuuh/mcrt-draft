#pragma once
#include "game_object.hpp"
#include "radiance_grid.hpp"

#include "glm/glm.hpp"

#include <vector>
#include <map>

namespace mcrt {
	struct Texture {
		~Texture()
		{
			if (pixel) delete[] pixel;
		}
		uint32_t* pixel{ nullptr };
		glm::ivec2 resolution{ -1 };
	};

	class Scene
	{
	public:
		Scene();

		glm::vec3 maxCoord() { return sceneMax; };
		glm::vec3 minCoord() { return sceneMin; };
		int numObjects() { return gameObjects.size(); };
		int amountVertices();
		std::vector<GameObject>& getGameObjects() { return gameObjects; };
		std::vector<std::shared_ptr<Texture>>& getTextures() { return textures; };

		void addGameObject(glm::vec3 position, glm::vec3 rotation, glm::vec3 scale, std::shared_ptr<Model> model);
		void loadModelFromOBJ(const std::string& fileName);
		int loadTexture(std::map<std::string, int>& knownTextures, const std::string& inFileName, const std::string& modelPath);

		void buildRadianceGrid(float cellSize);
		void normalize();

		RadianceGrid grid;
	private:
		std::vector<GameObject> gameObjects;
		std::vector<std::shared_ptr<Texture>> textures;


		glm::vec3 sceneMax;
		glm::vec3 sceneMin;
	};

}

