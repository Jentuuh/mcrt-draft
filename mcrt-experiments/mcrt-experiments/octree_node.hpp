#pragma once
#include "scene.hpp"
#include "glm/glm.hpp"
#include <vector>
#include <memory>
#include <stack>

namespace mcrt {
	class OctreeNode
	{
	public:
		OctreeNode(glm::vec3 min, glm::vec3 max, int octreeTextureRes, std::vector<float>&gpuOctree, std::stack<glm::vec3>& parentCoordStack, glm::ivec3& currentIndirectGridCoord,  int* nodeCount);

		void recursiveSplit(int currentLevel, int maxDepth, Scene& sceneObject, std::vector<float>& gpuOctree, std::stack<glm::vec3>& parentCoordStack, glm::ivec3& currentIndirectGridCoord, int* nodeCount);

	private:
		void updateCurrentIndirectionGridCoord(glm::ivec3& currentCoord);
		int octreeTextureRes;
		glm::vec3 min;
		glm::vec3 max;
	};
}


