#include "voxelizer.hpp"
#include "intersection_tester.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <math.h>
#include <stdio.h>
#include <iostream>

namespace mcrt {

	Voxelizer::Voxelizer(float voxelSize, std::shared_ptr<GameObject> object): voxelSize{voxelSize}, voxelizedObject{object}{}


	void Voxelizer::createBoundingBoxVoxelGrid()
	{
		AABB objBoundingBox = voxelizedObject->getWorldAABB();
		glm::vec3 difference = objBoundingBox.max - objBoundingBox.min;
		glm::ivec3 resolutions = { int(std::ceilf(difference.x / voxelSize)) + 1, int(std::ceilf(difference.y / voxelSize)) + 1, int(std::ceilf(difference.z / voxelSize)) + 1 };

		for (int x = 0; x < resolutions.x; x++)
		{
			for (int y = 0; y < resolutions.y; y++)
			{
				for (int z = 0; z < resolutions.z; z++)
				{
					bbVoxelGrid.push_back(Voxel{ {x,y,z}, voxelSize, objBoundingBox.min });
				}
			}
		}
	}

	void Voxelizer::voxelize()
	{
		createBoundingBoxVoxelGrid();
		std::vector<glm::vec3> vertices = voxelizedObject->getWorldVertices();

		// Loop through each bounding box voxel
		for (auto& v : bbVoxelGrid)
		{
			// Loop through object's triangles
			for (auto& t : voxelizedObject->model->mesh->indices)
			{
				// Triangle vertices
				glm::vec3 v0 = vertices[t.x];
				glm::vec3 v1 = vertices[t.y];
				glm::vec3 v2 = vertices[t.z];
				float* triangleVerts[3];
				triangleVerts[0] = glm::value_ptr(v0);
				triangleVerts[1] = glm::value_ptr(v1);
				triangleVerts[2] = glm::value_ptr(v2);


				glm::vec3 boxHalfSize = 0.5f * (v.max - v.min);
				glm::vec3 boxCenter = v.min + boxHalfSize;

				// If the triangle overlaps with the voxel, enable the voxel
				if (IntersectionTester::triangleBoxOverlap(glm::value_ptr(boxCenter), glm::value_ptr(boxHalfSize), triangleVerts))
				{
					// Add logical voxel
					resultVoxelGrid.push_back(v);
					break;
				}
			}
		}
		std::cout << "Voxelizing of object done. Went from " << vertices.size() << " vertices to " << resultVoxelGrid.size() << " voxels." << std::endl;
	}
}