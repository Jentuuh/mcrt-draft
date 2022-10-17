#pragma once
#include "glm/glm.hpp"
#include "model.hpp"
#include <memory>

namespace mcrt {
	struct Transform {
		Transform(glm::vec3 pos, glm::vec3 rot, glm::vec3 scale);

		void updatePosition(glm::vec3 newPos);
		void translate(glm::vec3 p_translation);
		void updateRotation(glm::vec3 newRot);
		void applySceneRescale(glm::vec3 p_scale);
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
		int amountVertices() { return model->vertices.size(); };

		std::shared_ptr<Model> model;
		Transform worldTransform;
	};
}


