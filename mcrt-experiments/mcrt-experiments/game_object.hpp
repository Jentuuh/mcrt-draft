#pragma once
#include "glm/glm.hpp"
#include "model.hpp"
#include <memory>

namespace mcrt {
	struct Transform {
		Transform(glm::vec3 pos, glm::vec3 rot, glm::vec3 scale);

		void updatePosition(glm::vec3 newPos);
		void updateRotation(glm::vec3 newRot);
		void updateScale(glm::vec3 newScale);
		glm::mat4x4 transformation();

		glm::vec3 translation;
		glm::vec3 rotation;
		glm::vec3 scale;
		glm::mat4x4 object2World;
	};

	class GameObject
	{
	public:
		GameObject(Transform transform, std::shared_ptr<Model> model);

		std::vector<glm::vec3> getWorldVertices();

		std::shared_ptr<Model> model;
		Transform worldTransform;
	};
}


