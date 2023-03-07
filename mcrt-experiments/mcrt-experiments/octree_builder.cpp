#include "octree_builder.hpp"
#include <iostream>
#include <fstream>

namespace mcrt {
	OctreeBuilder::OctreeBuilder(int maxDepth, int octreeTextureRes, Scene& sceneObject)
	{	std::cout << "==========================" << std::endl;
		std::cout << "    BUILDING OCTREE...    " << std::endl;
		std::cout << "==========================" << std::endl;

		// Data initialization
		gpuOctree.resize(octreeTextureRes * octreeTextureRes * octreeTextureRes, 0.0f);	// We allocate the maximum amount of space so we only need to make 1 allocation
		currentCoord = glm::ivec3{ 0, 0, 0};
		parentCoordStack.push(glm::vec3{0, 0, 0});
		numNodes = 0;

		// Make root and start recursive splitting
		root = std::make_unique<OctreeNode>(glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{ 1.0f, 1.0f, 1.0f },octreeTextureRes,gpuOctree, parentCoordStack, currentCoord, &numNodes);
		bool doesRootIntersect = OctreeNode::boxHasIntersections(glm::vec3{ 0.0f, 0.0f, 0.0f }, glm::vec3{ 1.0f, 1.0f, 1.0f }, sceneObject);
		root->recursiveSplit(0, maxDepth, sceneObject, gpuOctree, parentCoordStack, currentCoord, doesRootIntersect, &numNodes);

		// After build report
		int totalOctreeSize = currentCoord.z * octreeTextureRes * octreeTextureRes + currentCoord.y * octreeTextureRes + currentCoord.x * 32;
		std::cout << "Total octree size in floats: " << totalOctreeSize << std::endl;

		//int octreeAllocationSize = currentCoord.z > 0 ? octreeTextureRes * octreeTextureRes * (currentCoord.z + 1) : octreeTextureRes * (currentCoord.y + 1);
		//std::cout << "Octree allocation size (amount of floats): " << octreeAllocationSize << std::endl; // Resize the octree to conform to the CUDA 3D texture allocator (and to save as much space as possible)

		gpuOctree.resize(totalOctreeSize);
		initLeafPositions();

		std::cout << "Octree building done. Total number of nodes: " << numNodes << std::endl;
	}

	void OctreeBuilder::initLeafPositions()
	{
		root->pushLeafPositions(leafPositions);
		std::cout << "Leaf positions initialized. Length: " << leafPositions.size() << std::endl;
	}

	// For debugging purposes
	void OctreeBuilder::writeGPUOctreeToTxtFile(std::string filePath)
	{
		std::ofstream outputFile;
		outputFile.open(filePath);
		for (int n = 0; n < numNodes; n++)
		{
			for (int i = 0; i < 8; i++)
			{
				outputFile << "(";
				for (int j = 0; j < 4; j++)
				{
					outputFile << gpuOctree[n * 32 + i * 4 + j] << ",";
				}
				outputFile << ") ; ";
			}
			outputFile << "\n";
		}
	}

}