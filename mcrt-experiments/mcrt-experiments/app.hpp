#pragma once
#include "renderer.hpp"
#include "scene.hpp"
#include "GLFWindow.hpp"

#include <GL/gl.h>

#include <memory>


namespace mcrt {

    struct MCRTWindow : public GLFCameraWindow
    {
        MCRTWindow(const std::string& title,
            Scene& model,
            Camera& camera,
            const float worldScale)
            : GLFCameraWindow(title, camera.position, camera.target, camera.up, worldScale),
            sample(model, camera, UNBIASED, NA, TEXTURE_2D)
        {
            viewerObject = std::make_unique<GameObject>(Transform{ {0.5f, 0.5f, 0.5f}, {0.0f, glm::pi<float>(), 0.0f}, {1.0f, 1.0f, 1.0f}}, nullptr);
            //sample.updateCameraInCircle(viewerObject.get(), 0.0f);
            sample.updateCamera(viewerObject.get());
        }

        virtual void render(float deltaTime) override
        {
            
            cameraController.moveInPlaneXZ(this->handle, deltaTime, viewerObject.get());
            if (cameraController.cameraModified) {
                sample.updateCamera(viewerObject.get());
                cameraController.cameraModified = false;
            }

            //std::cout << "Frame composition time: " << deltaTime << " ms." << std::endl;
            //sample.updateCameraInCircle(viewerObject.get(), deltaTime);

            sample.render();
        }

        virtual void draw() override
        {
            sample.downloadPixels(pixels.data());
            if (fbTexture == 0)
                glGenTextures(1, &fbTexture);

            glBindTexture(GL_TEXTURE_2D, fbTexture);
            GLenum texFormat = GL_RGBA;
            GLenum texelType = GL_UNSIGNED_BYTE;
            glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
                texelType, pixels.data());

            glDisable(GL_LIGHTING);
            glColor3f(1, 1, 1);

            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();

            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, fbTexture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            glDisable(GL_DEPTH_TEST);

            glViewport(0, 0, fbSize.x, fbSize.y);

            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

            glBegin(GL_QUADS);
            {
                glTexCoord2f(0.f, 0.f);
                glVertex3f(0.f, 0.f, 0.f);

                glTexCoord2f(0.f, 1.f);
                glVertex3f(0.f, (float)fbSize.y, 0.f);

                glTexCoord2f(1.f, 1.f);
                glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

                glTexCoord2f(1.f, 0.f);
                glVertex3f((float)fbSize.x, 0.f, 0.f);
            }
            glEnd();
        }

        virtual void resize(const glm::ivec2& newSize)
        {
            fbSize = newSize;
            sample.resize(newSize, viewerObject.get());
            pixels.resize(newSize.x * newSize.y);
        }

        std::unique_ptr<GameObject> viewerObject;
        KeyboardMovementController cameraController;
        glm::ivec2                  fbSize;
        GLuint                      fbTexture{ 0 };
        Renderer                    sample;
        std::vector<uint32_t>       pixels;
    };

	class App
	{
	public:
		App();

		void run();
	private:
        void loadScene();

        std::unique_ptr<MCRTWindow> window;
        Scene scene;
	};
}

