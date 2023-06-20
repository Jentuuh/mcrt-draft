#include "textureSeams.cuh"

__global__ void remove_seams_kernel(void) {

}

namespace TextureSeams {
	void removeSeams(void)
	{
		remove_seams_kernel <<<1,1>>> ();
		printf("Hello I am in CUDA code!");
	}
}