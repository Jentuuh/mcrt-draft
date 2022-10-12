#include "renderer.hpp"
#include "GLFWindow.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
// std
#include <iostream>


namespace mcrt {
    struct MCRTWindow: public GLFWindow 
    {
        MCRTWindow(const std::string& title)
            : GLFWindow(title)
        {}

        virtual void render() override
        {
            sample.render();
        }

        virtual void draw() override
        {
            std::cout << "Test" << std::endl;
        }

        virtual void resize(const glm::ivec2& newSize)
        {
            fbSize = newSize;
            sample.resize(newSize);
            pixels.resize(newSize.x * newSize.y);
        }


        glm::ivec2                  fbSize;
        //GLuint                  fbTexture{ 0 };
        Renderer                    sample;
        std::vector<uint32_t>       pixels;
    };

	// Main entry point
	extern "C" int main(int argc, char* argv[]) {
        try {
            //Renderer sample;

            //// Resize framebuffer
            //const glm::vec2 fbSize{ 1200, 1024 };
            //sample.resize(fbSize);
            //sample.render();

            //// Download framebuffer from device
            //std::vector<uint32_t> pixels(fbSize.x * fbSize.y);
            //sample.downloadPixels(pixels.data());

            //// Save framebuffer image to file
            //const std::string fileName = "mcrt_test.png";
            //stbi_write_png(fileName.c_str(), fbSize.x, fbSize.y, 4,
            //    pixels.data(), fbSize.x * sizeof(uint32_t));
            //std::cout << std::endl
            //    << "Image rendered, and saved to " << fileName << " ... done." << std::endl
            //    << std::endl;
            MCRTWindow* window = new MCRTWindow("Optix 7 Course Example");
            window->run();
        }
        catch (std::runtime_error& e) {
            std::cout << "FATAL ERROR: " << e.what() << std::endl;
            exit(1);
        }
        return 0;
	}
}
