#include "radiance_cell.hpp"
#include "glm/gtc/matrix_transform.hpp"

namespace mcrt {
	RadianceCell::RadianceCell(glm::ivec3 coord, float scale) 
	{
        glm::mat4 scaleM = glm::scale(glm::mat4(1.0f), glm::vec3{ scale, scale, scale });
        glm::mat4 transM = glm::translate(glm::mat4(1.0f), glm::vec3{ scale * coord.x, scale * coord.y, scale * coord.z });

        glm::mat4 transform = transM * scaleM;

        // Transform the cell's vertices into the right position of the grid
        vertices.push_back(transform * glm::vec4{ 0.0f, 0.0f, 0.0f, 1.0f });
        vertices.push_back(transform * glm::vec4{ 1.0f, 0.0f, 0.0f, 1.0f });
        vertices.push_back(transform * glm::vec4{ 0.0f, 1.0f, 0.0f, 1.0f });
        vertices.push_back(transform * glm::vec4{ 1.0f, 1.0f, 0.0f, 1.0f });
        vertices.push_back(transform * glm::vec4{ 0.0f, 0.0f, 1.0f, 1.0f });
        vertices.push_back(transform * glm::vec4{ 1.0f, 0.0f, 1.0f, 1.0f });
        vertices.push_back(transform * glm::vec4{ 0.0f, 1.0f, 1.0f, 1.0f });
        vertices.push_back(transform * glm::vec4{ 1.0f, 1.0f, 1.0f, 1.0f });

        int indicesCube[] = { 0,1,3, 2,3,0,
                         5,7,6, 5,6,4,
                         0,4,5, 0,5,1,
                         2,3,7, 2,7,6,
                         1,5,7, 1,7,3,
                         4,0,2, 4,2,6
        };

        for (int i = 0; i < 12; i++)
            indices.push_back(glm::ivec3(indicesCube[3 * i + 0],
                indicesCube[3 * i + 1],
                indicesCube[3 * i + 2]));
	}

	void RadianceCell::addObject(std::shared_ptr<GameObject> obj) 
	{
		objectsInside.push_back(obj);
	}

	void RadianceCell::removeObject(std::shared_ptr<GameObject> obj)
	{
		remove(objectsInside.begin(), objectsInside.end(), obj);
	}
}