#pragma once
#include "game_object.hpp"
#include "radiance_grid.hpp"
#include "area_light.hpp"
#include "voxelizer.hpp"

#include <glm/glm.hpp>

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

	struct HDRTexture {
		~HDRTexture()
		{
			if (pixel) delete[] pixel;
		}
		float* pixel{ nullptr };
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
		int amountLights() { return lights.size(); };
		std::vector<std::shared_ptr<GameObject>>& getGameObjects() { return gameObjects; };
		std::vector<std::shared_ptr<Voxelizer>>& getVoxelizers() { return voxelizers; };
		std::vector<std::shared_ptr<Texture>>& getDiffuseTextures() { return diffuseTextures; };
		std::vector<std::shared_ptr<HDRTexture>>& getWorldPosTextures() { return worldPosTextures; };
		std::vector<std::shared_ptr<HDRTexture>>& getWorldNormalTextures() { return worldNormalTextures; };
		std::vector<std::shared_ptr<HDRTexture>>& getWorldDiffuseCoordsTextures() { return worldDiffuseCoordsTextures; };
		std::vector<LightData> getLightsData();

		void loadSponzaComponents();
		void loadWorldDataTextures();
		void addGameObject(glm::vec3 position, glm::vec3 rotation, glm::vec3 scale, std::shared_ptr<Model> model);
		void loadModelFromOBJ(const std::string& fileName);
		void loadModelFromOBJWithMultipleUVs(const std::string& fileNameLightMappedUVs, const std::string& fileNameTextureUVs);
		int loadTexture(std::map<std::string, int>& knownTextures, const std::string& inFileName, const std::string& modelPath);
		void loadDataTexture(const std::string& filePath, std::vector<std::shared_ptr<Texture>>& storageVector);
		void loadHDRDataTexture(const std::string& filePath, std::vector<std::shared_ptr<HDRTexture>>& storageVector);

		
		void voxelizeObjects();
		void buildRadianceGrid(float cellSize);
		void normalize();
		void loadLights();

		RadianceGrid grid;
	private:
		std::vector<std::shared_ptr<Voxelizer>> voxelizers;
		std::vector<std::shared_ptr<GameObject>> gameObjects;
		std::vector<std::shared_ptr<Texture>> diffuseTextures;
		std::vector<std::shared_ptr<HDRTexture>> worldPosTextures;
		std::vector<std::shared_ptr<HDRTexture>> worldNormalTextures;
		std::vector<std::shared_ptr<HDRTexture>> worldDiffuseCoordsTextures;
		std::vector<AreaLight> lights;

		glm::vec3 sceneMax;
		glm::vec3 sceneMin;
	};

}

