#include "renderer.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "GLFWindow.hpp"
#include <GL/gl.h>
#include <glm/gtx/string_cast.hpp>
// std
#include <iostream>


namespace mcrt {
    struct MCRTWindow: public GLFCameraWindow
    {
        MCRTWindow(const std::string& title,
                    const TriangleMesh& model,
                    const Camera& camera,
                    const float worldScale)
            : GLFCameraWindow(title, camera.position, camera.target, camera.up, worldScale),
            sample(model, camera)
        {        
            sample.updateCamera(camera);
        }

        virtual void render() override
        {
            if (cameraFrame.modified) {

                sample.updateCamera(Camera{ cameraFrame.get_from(),
                                         cameraFrame.get_at(),
                                         cameraFrame.get_up() });
                cameraFrame.modified = false;
            }
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
            sample.resize(newSize);
            pixels.resize(newSize.x * newSize.y);
        }


        glm::ivec2                  fbSize;
        GLuint                      fbTexture{ 0 };
        Renderer                    sample;
        std::vector<uint32_t>       pixels;
    };

	// Main entry point
	extern "C" int main(int argc, char* argv[]) {
        try {
            TriangleMesh model;
            // 100x100 thin ground plane
            model.addCube(glm::vec3(0.f, -1.5f, 0.f), glm::vec3(10.0f, 0.1f, 10.0f));
            // a unit cube centered on top of that
            model.addCube(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(2.0f, 2.0f, 2.0f));

            Camera camera = Camera{ glm::vec3{ -10.f,5.f,-12.f}, glm::vec3{0.f,0.f,0.f}, glm::vec3{0.0f, 1.0f, 0.0f } };

            // something approximating the scale of the world, so the
            // camera knows how much to move for any given user interaction:
            const float worldScale = 10.f;

            MCRTWindow* window = new MCRTWindow("Memory Coherent Ray Tracing", model, camera, worldScale);
            window->run();

    /*        Renderer sample{model};
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
        catch (std::runtime_error& e) {
            std::cout << "FATAL ERROR: " << e.what() << std::endl;
            exit(1);
        }
        return 0;
	}
}
