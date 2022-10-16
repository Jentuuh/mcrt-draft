#include "app.hpp"
#include "glm/gtx/string_cast.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace mcrt {
	App::App()
	{
		Camera camera = Camera{ glm::vec3{ -10.f,5.f,-12.f}, glm::vec3{0.f,0.f,0.f}, glm::vec3{0.0f, 1.0f, 0.0f } };
		scene = Scene{};
		loadScene();

		// something approximating the scale of the world, so the
		// camera knows how much to move for any given user interaction:
		const float worldScale = 10.f;

		window = std::make_unique<MCRTWindow>("Memory Coherent Ray Tracing", scene, camera, worldScale);
	}

	void App::run()
	{
		window->run();

		/* Renderer sample{model};
		sample.setCamera(camera);

		const glm::ivec2 fbSize(glm::ivec2(1200, 1024));
		sample.resize(fbSize);
		sample.render();
		std::vector<uint32_t> pixels(fbSize.x * fbSize.y);
		sample.downloadPixels(pixels.data());

		const std::string fileName = "mcrt_test.png";
		stbi_write_png(fileName.c_str(), fbSize.x, fbSize.y, 4,
		pixels.data(), fbSize.x * sizeof(uint32_t));*/
	}


	void App::loadScene()
	{
		std::shared_ptr<Model> cubeModel = std::make_shared<Model>();
		scene.addGameObject(glm::vec3{ -0.5f,-0.5f,-0.5f }, glm::vec3{ 0.0f,0.0f,0.0f }, glm::vec3{ 1.0f, 1.0f, 1.0f }, cubeModel);
		scene.addGameObject(glm::vec3{ -5.0f,-1.5f, -5.0f }, glm::vec3{ 0.0f,0.0f,0.0f }, glm::vec3{ 10.0f, 0.1f, 10.0f }, cubeModel);

		std::cout << "Loaded scene: " << scene.vertices().size() << " vertices. Scene Max: " << glm::to_string(scene.maxCoord()) << " Scene Min: " << glm::to_string(scene.minCoord()) << std::endl;
	}

}