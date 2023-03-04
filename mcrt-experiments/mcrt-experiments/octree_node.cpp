#include "octree_node.hpp"
#include "intersection_tester.hpp"

#include <stdexcept>
#include <glm/gtc/type_ptr.hpp>

namespace mcrt {
	OctreeNode::OctreeNode(glm::vec3 min, glm::vec3 max, int octreeTextureRes, std::vector<float>& gpuOctree, std::stack<glm::vec3>& parentCoordStack, glm::ivec3& currentIndirectGridCoord, int* nodeCount) : min{ min }, max{ max }, octreeTextureRes{octreeTextureRes}
	{
		if (!(octreeTextureRes % 32 == 0))
		{
			throw std::runtime_error("Octree texture resolution needs to be a multiple of 32!");
		}
		*nodeCount = *nodeCount + 1;
	}
	
	void OctreeNode::updateCurrentIndirectionGridCoord(glm::ivec3& currentCoord)
	{
		int gridWidth = int(octreeTextureRes / 32);

		currentCoord.x = (currentCoord.x + 1) % gridWidth;			// There are `gridWidth` nodes (indirection grids) in a row
		if (currentCoord.x == 0)
		{
			currentCoord.y = (currentCoord.y + 1) % octreeTextureRes;	// There are `octreeTextureRes` rows
		}
		if (currentCoord.x == 0 && currentCoord.y == 0)
		{
			currentCoord.z = currentCoord.z + 1;
			if (currentCoord.z > octreeTextureRes)	// There are `octreeTextureRes` texture layers
			{
				throw std::runtime_error("Octree builder went beyond allocated memory for octree texture.");
			}
		}
	}


	void OctreeNode::recursiveSplit(int currentLevel, int maxDepth, Scene& sceneObject, std::vector<float>& gpuOctree, std::stack<glm::vec3>& parentCoordStack, glm::ivec3& currentIndirectGridCoord, int* nodeCount)
	{
		glm::vec3 currentNodeCoord = parentCoordStack.top();
		int cellOffset = currentNodeCoord.z * octreeTextureRes * octreeTextureRes + currentNodeCoord.y * octreeTextureRes + currentNodeCoord.x * int(octreeTextureRes / 32);

		if (currentLevel >= maxDepth)
		{
			parentCoordStack.pop();
			return;
		}


		// Loop through game objects
		for (auto& g : sceneObject.getGameObjects())
		{
			std::vector<glm::vec3> vertices = g->getWorldVertices();

			// Loop through object's triangles
			for (auto& t : g->model->mesh->indices)
			{
				// Triangle vertices
				glm::vec3 v0 = vertices[t.x];
				glm::vec3 v1 = vertices[t.y];
				glm::vec3 v2 = vertices[t.z];
				float* triangleVerts[3];
				triangleVerts[0] = glm::value_ptr(v0);
				triangleVerts[1] = glm::value_ptr(v1);
				triangleVerts[2] = glm::value_ptr(v2);


				glm::vec3 boxHalfSize = 0.5f * (max - min);
				glm::vec3 boxCenter = min + boxHalfSize;

				// If a triangle of the object intersects with the node's bounding box, we split up further
				if (IntersectionTester::triangleBoxOverlap(glm::value_ptr(boxCenter), glm::value_ptr(boxHalfSize), triangleVerts))
				{
					int currentChild = 0;
					// =======================================================
					//				  CHILD 1: Left, front, up
					// =======================================================
					glm::vec3 min1 = boxCenter - glm::vec3{ boxHalfSize.x, 0.0f, boxHalfSize.z };
					glm::vec3 max1 = boxCenter + glm::vec3{ 0.0f, boxHalfSize.y, 0.0f };
					std::unique_ptr<OctreeNode> child1 = std::make_unique<OctreeNode>(min1, max1, octreeTextureRes, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
					int childOffset = currentChild * 4;

					gpuOctree[cellOffset + childOffset + 0] = float(currentIndirectGridCoord.x);	// R (x index)
					gpuOctree[cellOffset + childOffset + 1] = float(currentIndirectGridCoord.y);	// G (y index)
					gpuOctree[cellOffset + childOffset + 2] = float(currentIndirectGridCoord.z);	// B (z index)
					gpuOctree[cellOffset + childOffset + 3] = 0.5f;	// A (type)

					// Push relative coordinate (index) of new child onto the stack, so we can access it in further recursion calls
					parentCoordStack.push(glm::vec3{ currentIndirectGridCoord.x, currentIndirectGridCoord.y, currentIndirectGridCoord.z });
					updateCurrentIndirectionGridCoord(currentIndirectGridCoord);

					child1->recursiveSplit(currentLevel + 1, maxDepth, sceneObject, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
					currentChild++;

					// =======================================================
					//				  CHILD 2: Right, front, up
					// =======================================================
					glm::vec3 min2 = boxCenter - glm::vec3{ 0.0f, 0.0f, boxHalfSize.z };
					glm::vec3 max2 = boxCenter + glm::vec3{ boxHalfSize.x, boxHalfSize.y, 0.0f };
					std::unique_ptr<OctreeNode> child2 = std::make_unique<OctreeNode>(min2, max2, octreeTextureRes, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
					childOffset = currentChild * 4;

					gpuOctree[cellOffset + childOffset + 0] = float(currentIndirectGridCoord.x);	// R (x index)
					gpuOctree[cellOffset + childOffset + 1] = float(currentIndirectGridCoord.y);	// G (y index)
					gpuOctree[cellOffset + childOffset + 2] = float(currentIndirectGridCoord.z);	// B (z index)
					gpuOctree[cellOffset + childOffset + 3] = 0.5f;	// A (type)

					// Push relative coordinate (index) of new child onto the stack, so we can access it in further recursion calls
					parentCoordStack.push(glm::vec3{ currentIndirectGridCoord.x, currentIndirectGridCoord.y, currentIndirectGridCoord.z });
					updateCurrentIndirectionGridCoord(currentIndirectGridCoord);

					child2->recursiveSplit(currentLevel + 1, maxDepth, sceneObject, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
					currentChild++;

					// =======================================================
					//				  CHILD 3: Left, front, bottom
					// =======================================================
					glm::vec3 min3 = boxCenter - boxHalfSize;
					glm::vec3 max3 = boxCenter;
					std::unique_ptr<OctreeNode> child3 = std::make_unique<OctreeNode>(min3, max3, octreeTextureRes, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
					childOffset = currentChild * 4;

					gpuOctree[cellOffset + childOffset + 0] = float(currentIndirectGridCoord.x);	// R (x index)
					gpuOctree[cellOffset + childOffset + 1] = float(currentIndirectGridCoord.y);	// G (y index)
					gpuOctree[cellOffset + childOffset + 2] = float(currentIndirectGridCoord.z);	// B (z index)
					gpuOctree[cellOffset + childOffset + 3] = 0.5f;	// A (type)
					
					// Push relative coordinate (index) of new child onto the stack, so we can access it in further recursion calls
					parentCoordStack.push(glm::vec3{ currentIndirectGridCoord.x, currentIndirectGridCoord.y, currentIndirectGridCoord.z });
					updateCurrentIndirectionGridCoord(currentIndirectGridCoord);

					child3->recursiveSplit(currentLevel + 1, maxDepth, sceneObject, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
					currentChild++;

					// =======================================================
					//				  CHILD 4: Right, front, bottom
					// =======================================================
					glm::vec3 min4 = boxCenter - glm::vec3{ 0.0f, boxHalfSize.y, boxHalfSize.z };
					glm::vec3 max4 = boxCenter + glm::vec3{ boxHalfSize.x, 0.0f, 0.0f };
					std::unique_ptr<OctreeNode> child4 = std::make_unique<OctreeNode>(min4, max4, octreeTextureRes, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
					childOffset = currentChild * 4;

					gpuOctree[cellOffset + childOffset + 0] = float(currentIndirectGridCoord.x);	// R (x index)
					gpuOctree[cellOffset + childOffset + 1] = float(currentIndirectGridCoord.y);	// G (y index)
					gpuOctree[cellOffset + childOffset + 2] = float(currentIndirectGridCoord.z);	// B (z index)
					gpuOctree[cellOffset + childOffset + 3] = 0.5f;	// A (type)

					// Push relative coordinate (index) of new child onto the stack, so we can access it in further recursion calls
					parentCoordStack.push(glm::vec3{ currentIndirectGridCoord.x, currentIndirectGridCoord.y, currentIndirectGridCoord.z });
					updateCurrentIndirectionGridCoord(currentIndirectGridCoord);

					child4->recursiveSplit(currentLevel + 1, maxDepth, sceneObject, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
					currentChild++;

					// =======================================================
					//				  CHILD 5: Left, back, up
					// =======================================================
					glm::vec3 min5 = boxCenter - glm::vec3{ boxHalfSize.x, 0.0f, 0.0f };
					glm::vec3 max5 = boxCenter + glm::vec3{ 0.0f, boxHalfSize.y, boxHalfSize.z };
					std::unique_ptr<OctreeNode> child5 = std::make_unique<OctreeNode>(min5, max5, octreeTextureRes, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
					childOffset = currentChild * 4;

					gpuOctree[cellOffset + childOffset + 0] = float(currentIndirectGridCoord.x);	// R (x index)
					gpuOctree[cellOffset + childOffset + 1] = float(currentIndirectGridCoord.y);	// G (y index)
					gpuOctree[cellOffset + childOffset + 2] = float(currentIndirectGridCoord.z);	// B (z index)
					gpuOctree[cellOffset + childOffset + 3] = 0.5f;	// A (type)

					// Push relative coordinate (index) of new child onto the stack, so we can access it in further recursion calls
					parentCoordStack.push(glm::vec3{ currentIndirectGridCoord.x, currentIndirectGridCoord.y, currentIndirectGridCoord.z });
					updateCurrentIndirectionGridCoord(currentIndirectGridCoord);

					child5->recursiveSplit(currentLevel + 1, maxDepth, sceneObject, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
					currentChild++;

					// =======================================================
					//				  CHILD 6: Right, back, up
					// =======================================================
					glm::vec3 min6 = boxCenter;
					glm::vec3 max6 = boxCenter + glm::vec3{ boxHalfSize.x, boxHalfSize.y, boxHalfSize.z };
					std::unique_ptr<OctreeNode> child6 = std::make_unique<OctreeNode>(min6, max6, octreeTextureRes, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
					childOffset = currentChild * 4;

					gpuOctree[cellOffset + childOffset + 0] = float(currentIndirectGridCoord.x);	// R (x index)
					gpuOctree[cellOffset + childOffset + 1] = float(currentIndirectGridCoord.y);	// G (y index)
					gpuOctree[cellOffset + childOffset + 2] = float(currentIndirectGridCoord.z);	// B (z index)
					gpuOctree[cellOffset + childOffset + 3] = 0.5f;	// A (type)

					// Push relative coordinate (index) of new child onto the stack, so we can access it in further recursion calls
					parentCoordStack.push(glm::vec3{ currentIndirectGridCoord.x, currentIndirectGridCoord.y, currentIndirectGridCoord.z });
					updateCurrentIndirectionGridCoord(currentIndirectGridCoord);

					child6->recursiveSplit(currentLevel + 1, maxDepth, sceneObject, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
					currentChild++;

					// =======================================================
					//				  CHILD 7: Left, back, bottom
					// =======================================================
					glm::vec3 min7 = boxCenter - glm::vec3{ boxHalfSize.x, boxHalfSize.y, 0.0f };
					glm::vec3 max7 = boxCenter + glm::vec3{ 0.0f, 0.0f, boxHalfSize.z };
					std::unique_ptr<OctreeNode> child7 = std::make_unique<OctreeNode>(min7, max7, octreeTextureRes, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
					childOffset = currentChild * 4;

					gpuOctree[cellOffset + childOffset + 0] = float(currentIndirectGridCoord.x);	// R (x index)
					gpuOctree[cellOffset + childOffset + 1] = float(currentIndirectGridCoord.y);	// G (y index)
					gpuOctree[cellOffset + childOffset + 2] = float(currentIndirectGridCoord.z);	// B (z index)
					gpuOctree[cellOffset + childOffset + 3] = 0.5f;	// A (type)

					// Push relative coordinate (index) of new child onto the stack, so we can access it in further recursion calls
					parentCoordStack.push(glm::vec3{ currentIndirectGridCoord.x, currentIndirectGridCoord.y, currentIndirectGridCoord.z });
					updateCurrentIndirectionGridCoord(currentIndirectGridCoord);

					child7->recursiveSplit(currentLevel + 1, maxDepth, sceneObject, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
					currentChild++;

					// =======================================================
					//				  CHILD 8: Right, back, bottom
					// =======================================================
					glm::vec3 min8 = boxCenter - glm::vec3{ 0.0f, boxHalfSize.y, 0.0f };
					glm::vec3 max8 = boxCenter + glm::vec3{ boxHalfSize.x, 0.0f, boxHalfSize.z };
					std::unique_ptr<OctreeNode> child8 = std::make_unique<OctreeNode>(min8, max8, octreeTextureRes, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
					childOffset = currentChild * 4;

					gpuOctree[cellOffset + childOffset + 0] = float(currentIndirectGridCoord.x);	// R (x index)
					gpuOctree[cellOffset + childOffset + 1] = float(currentIndirectGridCoord.y);	// G (y index)
					gpuOctree[cellOffset + childOffset + 2] = float(currentIndirectGridCoord.z);	// B (z index)
					gpuOctree[cellOffset + childOffset + 3] = 0.5f;	// A (type)

					// Push relative coordinate (index) of new child onto the stack, so we can access it in further recursion calls
					parentCoordStack.push(glm::vec3{ currentIndirectGridCoord.x, currentIndirectGridCoord.y, currentIndirectGridCoord.z });
					updateCurrentIndirectionGridCoord(currentIndirectGridCoord);

					child8->recursiveSplit(currentLevel + 1, maxDepth, sceneObject, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);

					parentCoordStack.pop();
					return;
				}
			}
		}
	}

}