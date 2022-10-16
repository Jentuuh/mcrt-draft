#include "model.hpp"

namespace mcrt {

	Model::Model()
	{
        loadModel();
	}

	void Model::loadModel()
	{
        int firstVertexID = (int)vertices.size();
        vertices.push_back(glm::vec4{ 0.0f, 0.0f, 0.0f, 1.0f });
        vertices.push_back(glm::vec4{ 1.0f, 0.0f, 0.0f, 1.0f });
        vertices.push_back(glm::vec4{ 0.0f, 1.0f, 0.0f, 1.0f });
        vertices.push_back(glm::vec4{ 1.0f, 1.0f, 0.0f, 1.0f });
        vertices.push_back(glm::vec4{ 0.0f, 0.0f, 1.0f, 1.0f });
        vertices.push_back(glm::vec4{ 1.0f, 0.0f, 1.0f, 1.0f });
        vertices.push_back(glm::vec4{ 0.0f, 1.0f, 1.0f, 1.0f });
        vertices.push_back(glm::vec4{ 1.0f, 1.0f, 1.0f, 1.0f });

        int indicesCube[] = { 0,1,3, 2,3,0,
                         5,7,6, 5,6,4,
                         0,4,5, 0,5,1,
                         2,3,7, 2,7,6,
                         1,5,7, 1,7,3,
                         4,0,2, 4,2,6
        };

        for (int i = 0; i < 12; i++)
            indices.push_back(firstVertexID + glm::ivec3(indicesCube[3 * i + 0],
                indicesCube[3 * i + 1],
                indicesCube[3 * i + 2]));
	}
}
