#include "octree_builder.hpp"
#include <iostream>

namespace mcrt {
	OctreeBuilder::OctreeBuilder(int maxDepth, int octreeTextureRes, Scene& sceneObject)
	{
		std::cout << "Building octree..." << std::endl;
		currentCoord = glm::ivec3{ 1, 0, 0};
		parentCoordStack.push(glm::vec3{0, 0, 0});
		numNodes = 0;
		root = std::make_unique<OctreeNode>(glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{ 1.0f, 1.0f, 1.0f },octreeTextureRes,gpuOctree, parentCoordStack, currentCoord, &numNodes);
		root->recursiveSplit(0, maxDepth, sceneObject, gpuOctree, parentCoordStack, currentCoord, &numNodes);
		std::cout << "Octree building done. Total number of nodes: " << numNodes << std::endl;
	}

}