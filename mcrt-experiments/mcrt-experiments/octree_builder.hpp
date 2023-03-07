#pragma once
#include "octree_node.hpp"
#include "scene.hpp"
#include <stack>
#include <memory>

namespace mcrt {
	class OctreeBuilder
	{
	public:
		OctreeBuilder(int maxDepth, int octreeTextureRes, Scene& sceneObject);

		void writeGPUOctreeToTxtFile(std::string filePath);
		void initLeafPositions();
		glm::ivec3 getTextureDimensions() { return currentCoord; };
		std::vector<float>& getOctree() { return gpuOctree; };

	private:
		std::vector<glm::vec3> leafPositions;
		std::unique_ptr<OctreeNode> root;
		std::stack<glm::vec3> parentCoordStack;
		std::vector<float> gpuOctree;
		glm::ivec3 currentCoord; // Maintains which indirection grid we are currently at (how many nodes we have already inserted (currentX, currentY, currentZ))
		int numNodes;
	};
}

