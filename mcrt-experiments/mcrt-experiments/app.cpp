#include "app.hpp"
#include "glm/gtx/string_cast.hpp"


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
	}


	void App::loadScene()
	{
		// ==============================
		//		   Scene loading
		// ==============================
		//scene.loadModelFromOBJ("../models/cornell/cornell.obj");
		//scene.loadCornellComponents();
		scene.loadSponzaComponents();
		//scene.loadRungholtComponents();

		std::cout << "Loaded scene: " << scene.amountVertices() << " vertices. Scene Max: " << glm::to_string(scene.maxCoord()) << " Scene Min: " << glm::to_string(scene.minCoord()) << std::endl;
		
		// Normalize scene to be contained within [0;1] in each dimension
		scene.normalize();
		// Create light sources
		scene.loadLights();
		// Build proxy geometry
		//scene.voxelizeObjects();


		// =======================================
		//		Algorithm data structure setup
		// =======================================
		// Build radiance grid that is contained within the scene
		scene.buildRadianceGrid(.5f);

		//// For each radiance cell, check which objects are (partially) in it
		//scene.grid.assignObjectsToCells(scene.getVoxelizers());
	}

}