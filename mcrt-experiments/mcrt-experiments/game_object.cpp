#include "game_object.hpp"
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>


namespace mcrt {
	Transform::Transform(glm::vec3 pos, glm::vec3 rot, glm::vec3 scale):translation{pos}, rotation{rot}, scale{scale}
	{
		// T*R*S
		object2World = transformation();
	}

	void Transform::updatePosition(glm::vec3 newPos)
	{
		translation = newPos;
		object2World = transformation();
	}

	void Transform::translate(glm::vec3 p_translation)
	{
		translation += p_translation;
		object2World = transformation();
	}

	void Transform::updateRotation(glm::vec3 newRot)
	{
		rotation = newRot;
		object2World = transformation();
	}

	void Transform::applySceneRescale(glm::vec3 p_scale)
	{
		glm::mat4x4 scaleM = glm::scale(glm::mat4x4(1.0f), p_scale);

		object2World = scaleM * object2World;
	}

	void Transform::updateScale(glm::vec3 newScale)
	{
		scale = newScale;
		object2World = transformation();
	}

	glm::mat4x4 Transform::transformation()
	{
		glm::mat4x4 rotationM = glm::mat4x4(1.0f);
		rotationM = glm::rotate(rotationM, rotation.x, glm::vec3{ 1.0f, 0.0f, 0.0f });
		rotationM = glm::rotate(rotationM, rotation.y, glm::vec3{ 0.0f, 1.0f, 0.0f });
		rotationM = glm::rotate(rotationM, rotation.z, glm::vec3{ 0.0f, 0.0f, 1.0f });

		glm::mat4x4 scaleM = glm::mat4x4(1.0f);
		scaleM = glm::scale(scaleM, scale);

		glm::mat4x4 transM = glm::mat4x4(1.0f);
		transM = glm::translate(transM, translation);

		return transM * rotationM * scaleM;
	}

	GameObject::GameObject(Transform transform, std::shared_ptr<Model> model):worldTransform{transform}
	{
		this->model = model;
	}

	std::vector<glm::vec3> GameObject::getWorldVertices()
	{
		std::vector<glm::vec3> worldVertices;
		for (auto& v : model->mesh->vertices)
		{
			worldVertices.push_back(worldTransform.object2World * glm::vec4{ v, 1.0f });
		}
		return worldVertices;
	}
}