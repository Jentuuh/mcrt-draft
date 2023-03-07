#include "octree_node.hpp"
#include "intersection_tester.hpp"

#include <iostream>
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
		bool updateY = false;
		bool updateZ = false;
		int gridWidth = int(octreeTextureRes / 32);

		currentCoord.x++;
		if (currentCoord.x == gridWidth) // There are `gridWidth` nodes (indirection grids) in a row
		{
			currentCoord.x = 0;
			updateY = true;
		}
		if (updateY)
		{
			currentCoord.y++;
			if (currentCoord.y == octreeTextureRes) // There are `octreeTextureRes` rows
			{
				currentCoord.y = 0;
				updateZ = true;
			}
		}
		if (updateZ)
		{
			currentCoord.z++;
			if (currentCoord.z >= octreeTextureRes) // There are `octreeTextureRes` texture layers
			{
				throw std::runtime_error("Octree builder went beyond allocated memory for octree texture.");
			}
		}
	}


	void OctreeNode::recursiveSplit(int currentLevel, int maxDepth, Scene& sceneObject, std::vector<float>& gpuOctree, std::stack<glm::vec3>& parentCoordStack, glm::ivec3& currentIndirectGridCoord, bool nodeIntersectsWithSceneGeometry, int* nodeCount)
	{
		// The location of the current node in the octree vector (necessary to calculate the offset into this vector)
		glm::vec3 currentNodeCoord = parentCoordStack.top();
		int cellOffset = currentNodeCoord.z * octreeTextureRes * octreeTextureRes + currentNodeCoord.y * octreeTextureRes + currentNodeCoord.x * 32;

		if (currentLevel >= maxDepth)
		{
			parentCoordStack.pop();
			return;
		}

		glm::vec3 boxHalfSize = 0.5f * (max - min);
		glm::vec3 boxCenter = min + boxHalfSize;

		// If a triangle of the object intersects with the node's bounding box, we split up further
		if (nodeIntersectsWithSceneGeometry)
		{
			// Update the indirection grid coordinate tracker
			updateCurrentIndirectionGridCoord(currentIndirectGridCoord);

			int currentChild = 0;
			// =======================================================
			//				  CHILD 1: Left, front, up
			// =======================================================
			glm::vec3 min1 = boxCenter - glm::vec3{ boxHalfSize.x, 0.0f, boxHalfSize.z };
			glm::vec3 max1 = boxCenter + glm::vec3{ 0.0f, boxHalfSize.y, 0.0f };
			std::shared_ptr<OctreeNode> child1 = std::make_shared<OctreeNode>(min1, max1, octreeTextureRes, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
			children.push_back(child1);
			int childOffset = currentChild * 4;
			bool doesChild1Intersect = boxHasIntersections(min1, max1, sceneObject);

			if (currentLevel + 1 < maxDepth && doesChild1Intersect) {
				gpuOctree[cellOffset + childOffset + 0] = float(currentIndirectGridCoord.x);	// R (x index)
				gpuOctree[cellOffset + childOffset + 1] = float(currentIndirectGridCoord.y);	// G (y index)
				gpuOctree[cellOffset + childOffset + 2] = float(currentIndirectGridCoord.z);	// B (z index)
				gpuOctree[cellOffset + childOffset + 3] = 0.5f;	// A (type)
			}
			else {
				children.push_back(nullptr);
			}

			// Push relative coordinate (index) of new child onto the stack, so we can access it in further recursion calls
			parentCoordStack.push(glm::vec3{ currentIndirectGridCoord.x, currentIndirectGridCoord.y, currentIndirectGridCoord.z });
			child1->recursiveSplit(currentLevel + 1, maxDepth, sceneObject, gpuOctree, parentCoordStack, currentIndirectGridCoord, doesChild1Intersect, nodeCount);
			currentChild++;

			// =======================================================
			//				  CHILD 2: Right, front, up
			// =======================================================
			glm::vec3 min2 = boxCenter - glm::vec3{ 0.0f, 0.0f, boxHalfSize.z };
			glm::vec3 max2 = boxCenter + glm::vec3{ boxHalfSize.x, boxHalfSize.y, 0.0f };
			std::shared_ptr<OctreeNode> child2 = std::make_shared<OctreeNode>(min2, max2, octreeTextureRes, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
			children.push_back(child2);
			childOffset = currentChild * 4;
			bool doesChild2Intersect = boxHasIntersections(min2, max2, sceneObject);

			if (currentLevel + 1 < maxDepth && doesChild2Intersect) {
				gpuOctree[cellOffset + childOffset + 0] = float(currentIndirectGridCoord.x);	// R (x index)
				gpuOctree[cellOffset + childOffset + 1] = float(currentIndirectGridCoord.y);	// G (y index)
				gpuOctree[cellOffset + childOffset + 2] = float(currentIndirectGridCoord.z);	// B (z index)
				gpuOctree[cellOffset + childOffset + 3] = 0.5f;	// A (type)
			}
			else {
				children.push_back(nullptr);
			}

			// Push relative coordinate (index) of new child onto the stack, so we can access it in further recursion calls
			parentCoordStack.push(glm::vec3{ currentIndirectGridCoord.x, currentIndirectGridCoord.y, currentIndirectGridCoord.z });
			child2->recursiveSplit(currentLevel + 1, maxDepth, sceneObject, gpuOctree, parentCoordStack, currentIndirectGridCoord, doesChild2Intersect, nodeCount);
			currentChild++;

			// =======================================================
			//				  CHILD 3: Left, front, bottom
			// =======================================================
			glm::vec3 min3 = boxCenter - boxHalfSize;
			glm::vec3 max3 = boxCenter;
			std::shared_ptr<OctreeNode> child3 = std::make_shared<OctreeNode>(min3, max3, octreeTextureRes, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
			children.push_back(child3);
			childOffset = currentChild * 4;
			bool doesChild3Intersect = boxHasIntersections(min3, max3, sceneObject);

			if (currentLevel + 1 < maxDepth && doesChild3Intersect) {
				gpuOctree[cellOffset + childOffset + 0] = float(currentIndirectGridCoord.x);	// R (x index)
				gpuOctree[cellOffset + childOffset + 1] = float(currentIndirectGridCoord.y);	// G (y index)
				gpuOctree[cellOffset + childOffset + 2] = float(currentIndirectGridCoord.z);	// B (z index)
				gpuOctree[cellOffset + childOffset + 3] = 0.5f;	// A (type)
			}
			else {
				children.push_back(nullptr);
			}
					
			// Push relative coordinate (index) of new child onto the stack, so we can access it in further recursion calls
			parentCoordStack.push(glm::vec3{ currentIndirectGridCoord.x, currentIndirectGridCoord.y, currentIndirectGridCoord.z });
			child3->recursiveSplit(currentLevel + 1, maxDepth, sceneObject, gpuOctree, parentCoordStack, currentIndirectGridCoord, doesChild3Intersect, nodeCount);
			currentChild++;

			// =======================================================
			//				  CHILD 4: Right, front, bottom
			// =======================================================
			glm::vec3 min4 = boxCenter - glm::vec3{ 0.0f, boxHalfSize.y, boxHalfSize.z };
			glm::vec3 max4 = boxCenter + glm::vec3{ boxHalfSize.x, 0.0f, 0.0f };
			std::shared_ptr<OctreeNode> child4 = std::make_shared<OctreeNode>(min4, max4, octreeTextureRes, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
			children.push_back(child4);
			childOffset = currentChild * 4;
			bool doesChild4Intersect = boxHasIntersections(min4, max4, sceneObject);

			if (currentLevel + 1 < maxDepth && doesChild4Intersect) {
				gpuOctree[cellOffset + childOffset + 0] = float(currentIndirectGridCoord.x);	// R (x index)
				gpuOctree[cellOffset + childOffset + 1] = float(currentIndirectGridCoord.y);	// G (y index)
				gpuOctree[cellOffset + childOffset + 2] = float(currentIndirectGridCoord.z);	// B (z index)
				gpuOctree[cellOffset + childOffset + 3] = 0.5f;	// A (type)
			}
			else {
				children.push_back(nullptr);
			}

			// Push relative coordinate (index) of new child onto the stack, so we can access it in further recursion calls
			parentCoordStack.push(glm::vec3{ currentIndirectGridCoord.x, currentIndirectGridCoord.y, currentIndirectGridCoord.z });
			child4->recursiveSplit(currentLevel + 1, maxDepth, sceneObject, gpuOctree, parentCoordStack, currentIndirectGridCoord, doesChild4Intersect, nodeCount);
			currentChild++;

			// =======================================================
			//				  CHILD 5: Left, back, up
			// =======================================================
			glm::vec3 min5 = boxCenter - glm::vec3{ boxHalfSize.x, 0.0f, 0.0f };
			glm::vec3 max5 = boxCenter + glm::vec3{ 0.0f, boxHalfSize.y, boxHalfSize.z };
			std::shared_ptr<OctreeNode> child5 = std::make_shared<OctreeNode>(min5, max5, octreeTextureRes, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
			children.push_back(child5);
			childOffset = currentChild * 4;
			bool doesChild5Intersect = boxHasIntersections(min5, max5, sceneObject);

			if (currentLevel + 1 < maxDepth && doesChild5Intersect) {
				gpuOctree[cellOffset + childOffset + 0] = float(currentIndirectGridCoord.x);	// R (x index)
				gpuOctree[cellOffset + childOffset + 1] = float(currentIndirectGridCoord.y);	// G (y index)
				gpuOctree[cellOffset + childOffset + 2] = float(currentIndirectGridCoord.z);	// B (z index)
				gpuOctree[cellOffset + childOffset + 3] = 0.5f;	// A (type)
			}
			else {
				children.push_back(nullptr);
			}

			// Push relative coordinate (index) of new child onto the stack, so we can access it in further recursion calls
			parentCoordStack.push(glm::vec3{ currentIndirectGridCoord.x, currentIndirectGridCoord.y, currentIndirectGridCoord.z });
			child5->recursiveSplit(currentLevel + 1, maxDepth, sceneObject, gpuOctree, parentCoordStack, currentIndirectGridCoord, doesChild5Intersect, nodeCount);
			currentChild++;

			// =======================================================
			//				  CHILD 6: Right, back, up
			// =======================================================
			glm::vec3 min6 = boxCenter;
			glm::vec3 max6 = boxCenter + glm::vec3{ boxHalfSize.x, boxHalfSize.y, boxHalfSize.z };
			std::shared_ptr<OctreeNode> child6 = std::make_shared<OctreeNode>(min6, max6, octreeTextureRes, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
			children.push_back(child6);
			childOffset = currentChild * 4;
			bool doesChild6Intersect = boxHasIntersections(min6, max6, sceneObject);

			if (currentLevel + 1 < maxDepth && doesChild6Intersect) {
				gpuOctree[cellOffset + childOffset + 0] = float(currentIndirectGridCoord.x);	// R (x index)
				gpuOctree[cellOffset + childOffset + 1] = float(currentIndirectGridCoord.y);	// G (y index)
				gpuOctree[cellOffset + childOffset + 2] = float(currentIndirectGridCoord.z);	// B (z index)
				gpuOctree[cellOffset + childOffset + 3] = 0.5f;	// A (type)
			}
			else {
				children.push_back(nullptr);
			}

			// Push relative coordinate (index) of new child onto the stack, so we can access it in further recursion calls
			parentCoordStack.push(glm::vec3{ currentIndirectGridCoord.x, currentIndirectGridCoord.y, currentIndirectGridCoord.z });
			child6->recursiveSplit(currentLevel + 1, maxDepth, sceneObject, gpuOctree, parentCoordStack, currentIndirectGridCoord, doesChild6Intersect, nodeCount);
			currentChild++;

			// =======================================================
			//				  CHILD 7: Left, back, bottom
			// =======================================================
			glm::vec3 min7 = boxCenter - glm::vec3{ boxHalfSize.x, boxHalfSize.y, 0.0f };
			glm::vec3 max7 = boxCenter + glm::vec3{ 0.0f, 0.0f, boxHalfSize.z };
			std::shared_ptr<OctreeNode> child7 = std::make_shared<OctreeNode>(min7, max7, octreeTextureRes, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
			children.push_back(child7);
			childOffset = currentChild * 4;
			bool doesChild7Intersect = boxHasIntersections(min7, max7, sceneObject);

			if (currentLevel + 1 < maxDepth && doesChild7Intersect) {
				gpuOctree[cellOffset + childOffset + 0] = float(currentIndirectGridCoord.x);	// R (x index)
				gpuOctree[cellOffset + childOffset + 1] = float(currentIndirectGridCoord.y);	// G (y index)
				gpuOctree[cellOffset + childOffset + 2] = float(currentIndirectGridCoord.z);	// B (z index)
				gpuOctree[cellOffset + childOffset + 3] = 0.5f;	// A (type)
			}
			else {
				children.push_back(nullptr);
			}

			// Push relative coordinate (index) of new child onto the stack, so we can access it in further recursion calls
			parentCoordStack.push(glm::vec3{ currentIndirectGridCoord.x, currentIndirectGridCoord.y, currentIndirectGridCoord.z });
			child7->recursiveSplit(currentLevel + 1, maxDepth, sceneObject, gpuOctree, parentCoordStack, currentIndirectGridCoord, doesChild7Intersect, nodeCount);
			currentChild++;

			// =======================================================
			//				  CHILD 8: Right, back, bottom
			// =======================================================
			glm::vec3 min8 = boxCenter - glm::vec3{ 0.0f, boxHalfSize.y, 0.0f };
			glm::vec3 max8 = boxCenter + glm::vec3{ boxHalfSize.x, 0.0f, boxHalfSize.z };
			std::shared_ptr<OctreeNode> child8 = std::make_shared<OctreeNode>(min8, max8, octreeTextureRes, gpuOctree, parentCoordStack, currentIndirectGridCoord, nodeCount);
			children.push_back(child8);
			childOffset = currentChild * 4;
			bool doesChild8Intersect = boxHasIntersections(min8, max8, sceneObject);

			if (currentLevel + 1 < maxDepth && doesChild8Intersect) {
				gpuOctree[cellOffset + childOffset + 0] = float(currentIndirectGridCoord.x);	// R (x index)
				gpuOctree[cellOffset + childOffset + 1] = float(currentIndirectGridCoord.y);	// G (y index)
				gpuOctree[cellOffset + childOffset + 2] = float(currentIndirectGridCoord.z);	// B (z index)
				gpuOctree[cellOffset + childOffset + 3] = 0.5f;	// A (type)
			}
			else {
				children.push_back(nullptr);
			}

			// Push relative coordinate (index) of new child onto the stack, so we can access it in further recursion calls
			parentCoordStack.push(glm::vec3{ currentIndirectGridCoord.x, currentIndirectGridCoord.y, currentIndirectGridCoord.z });
			child8->recursiveSplit(currentLevel + 1, maxDepth, sceneObject, gpuOctree, parentCoordStack, currentIndirectGridCoord, doesChild8Intersect, nodeCount);

			parentCoordStack.pop();
			return;
		}

	}

	bool OctreeNode::boxHasIntersections(glm::vec3 minBox, glm::vec3 maxBox, Scene& sceneObject)
	{
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


				glm::vec3 boxHalfSize = 0.5f * (maxBox - minBox);
				glm::vec3 boxCenter = minBox + boxHalfSize;

				// If a triangle of the object intersects with the node's bounding box, we split up further
				if (IntersectionTester::triangleBoxOverlap(glm::value_ptr(boxCenter), glm::value_ptr(boxHalfSize), triangleVerts))
				{
					return true;
				}
			}
		}
		return false;
	}	

	void OctreeNode::pushLeafPositions(std::vector<glm::vec3>& leafPositionsVector)
	{
		if (children.size() > 0)
		{
			// If one of the children is a nullptr they should all be
			if (children[0] == nullptr && children[1] == nullptr && children[2] == nullptr && children[3] == nullptr && children[4] == nullptr && children[5] == nullptr && children[6] == nullptr && children[7] == nullptr)
			{
				// If the node has no children it is a leaf node, so we push its center to the leafPositions vector.
				leafPositionsVector.push_back(getCenter());
			}
			else {
				// Otherwise, we loop over its children and call this function recursively.
				for (int i = 0; i < 8; i++)
				{
					children[i]->pushLeafPositions(leafPositionsVector);
				}
			}
		}
	}



}