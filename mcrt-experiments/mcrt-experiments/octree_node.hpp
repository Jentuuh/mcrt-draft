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

		void recursiveSplit(int currentLevel, int maxDepth, Scene& sceneObject, std::vector<float>& gpuOctree, std::stack<glm::vec3>& parentCoordStack, glm::ivec3& currentIndirectGridCoord, bool nodeIntersectsWithSceneGeometry, int* nodeCount);
		static bool boxHasIntersections(glm::vec3 minBox, glm::vec3 maxBox, Scene& sceneObject);
		void pushLeafPositions(std::vector<glm::vec3>& leafPositionsVector);

		std::vector<std::shared_ptr<OctreeNode>>& getChildren() { return children; };
		glm::vec3 getCenter() {return (max - min) / 2.0f; };

	private:
		void updateCurrentIndirectionGridCoord(glm::ivec3& currentCoord);
		int octreeTextureRes;
		glm::vec3 min;
		glm::vec3 max;
		std::vector<std::shared_ptr<OctreeNode>> children;
	};
}


