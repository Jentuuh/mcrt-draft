#include "octree_builder.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

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
		progress = 0.0f;

		// Make root and start recursive splitting
		root = std::make_unique<OctreeNode>(glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{ 1.0f, 1.0f, 1.0f },octreeTextureRes,gpuOctree, parentCoordStack, currentCoord, &numNodes);
		bool doesRootIntersect = OctreeNode::boxHasIntersections(glm::vec3{ 0.0f, 0.0f, 0.0f }, glm::vec3{ 1.0f, 1.0f, 1.0f }, sceneObject);

		auto increaseProgressCallback = std::bind(&OctreeBuilder::increaseProgress, this, std::placeholders::_1);
		root->recursiveSplit(0, maxDepth, sceneObject, gpuOctree, parentCoordStack, currentCoord, doesRootIntersect, &numNodes, increaseProgressCallback);

		// After build report
		int totalOctreeSize = currentCoord.z * octreeTextureRes * octreeTextureRes + currentCoord.y * octreeTextureRes + currentCoord.x * 32;
		std::cout << "Total octree size in floats: " << totalOctreeSize << ". Total size in MB: " << ((totalOctreeSize * 4) / 10e6) << " MB." << std::endl;

		// Save unnecessary space
		gpuOctree.resize(totalOctreeSize);

		std::cout << "Octree building done. Total number of nodes: " << numNodes << std::endl;
	}

	// Constructor that loads from file
	OctreeBuilder::OctreeBuilder(std::string loadOctreePath)
	{
		std::ifstream octreeFile; 
		octreeFile.open(loadOctreePath);

		std::string line;
		while (std::getline(octreeFile, line))
		{
			std::istringstream iss(line);
			float nextEntry;
			if (!(iss >> nextEntry)) { break; } // error
			gpuOctree.push_back(nextEntry);
		}

		std::cout << "Loaded octree. Total octree size in floats: " << gpuOctree.size() << ". Total size in MB: " << ((gpuOctree.size() * 4) / 10e6) << " MB." << std::endl;
	}

	// Save built octree to file
	void OctreeBuilder::saveOctreeToFile(std::string filePath)
	{
		std::ofstream outputFile;
		outputFile.open(filePath);
		for (int e = 0; e < gpuOctree.size(); e++)
		{
			outputFile << gpuOctree[e] << "\n";
		}
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

	void OctreeBuilder::increaseProgress(int reportDepthLevel)
	{
		progress += 1 / powf(8, reportDepthLevel);
		std::cout << "Octree building progress: " << (progress / 1.0f) << "%." << std::endl;
	}


}