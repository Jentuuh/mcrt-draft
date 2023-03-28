"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" -ptx mcrt-experiments/cameraRayProgram.cu -I "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.5.0\include" -I "3rdparty/glm" -I "3rdparty/nvidia" --generate-line-info -use_fast_math -o mcrt-experiments/devicePrograms.ptx
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\bin2c.exe" -c --padd 0 --type char --name embedded_ptx_code mcrt-experiments/devicePrograms.ptx > mcrt-experiments/devicePrograms_embedded.c 

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" -ptx mcrt-experiments/cameraRayProgramOctree.cu -I "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.5.0\include" -I "3rdparty/glm" -I "3rdparty/nvidia" --generate-line-info -use_fast_math -o mcrt-experiments/deviceProgramsOctree.ptx
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\bin2c.exe" -c --padd 0 --type char --name embedded_ptx_code_octree mcrt-experiments/deviceProgramsOctree.ptx > mcrt-experiments/devicePrograms_embedded_octree.c 

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" -ptx mcrt-experiments/directLightingShadersGathering.cu -I "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.5.0\include" -I "3rdparty/glm" -I "3rdparty/nvidia" --generate-line-info -use_fast_math -o mcrt-experiments/directLightingShadersGathering.ptx
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\bin2c.exe" -c --padd 0 --type char --name embedded_ptx_code_direct_lighting_gathering mcrt-experiments/directLightingShadersGathering.ptx > mcrt-experiments/directLightingShadersGathering_embedded.c 

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" -ptx mcrt-experiments/directLightingShadersGatheringOctree.cu -I "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.5.0\include" -I "3rdparty/glm" -I "3rdparty/nvidia" --generate-line-info -use_fast_math -o mcrt-experiments/directLightingShadersGatheringOctree.ptx
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\bin2c.exe" -c --padd 0 --type char --name embedded_ptx_code_direct_lighting_gathering_octree mcrt-experiments/directLightingShadersGatheringOctree.ptx > mcrt-experiments/directLightingShadersGathering_embedded_octree.c 

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" -ptx mcrt-experiments/radianceCellGathering_Cubemap3.cu -I "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.5.0\include" -I "3rdparty/glm" -I "3rdparty/nvidia" --generate-line-info -use_fast_math -o mcrt-experiments/radianceCellGathering_Cubemap.ptx
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\bin2c.exe" -c --padd 0 --type char --name embedded_ptx_code_radiance_cell_gathering_cubemap mcrt-experiments/radianceCellGathering_Cubemap.ptx > mcrt-experiments/radianceCellGathering_Cubemap_embedded.c

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" -ptx mcrt-experiments/radianceCellGathering_Cubemap_Octree.cu -I "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.5.0\include" -I "3rdparty/glm" -I "3rdparty/nvidia" --generate-line-info -use_fast_math -o mcrt-experiments/radianceCellGathering_Cubemap_Octree.ptx
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\bin2c.exe" -c --padd 0 --type char --name embedded_ptx_code_radiance_cell_gathering_cubemap_octree mcrt-experiments/radianceCellGathering_Cubemap_Octree.ptx > mcrt-experiments/radianceCellGathering_Cubemap_embedded_octree.c 

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" -ptx mcrt-experiments/radianceCellScattering_Cubemap_hybrid.cu -I "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.5.0\include" -I "3rdparty/glm" -I "3rdparty/nvidia" --generate-line-info -use_fast_math -o mcrt-experiments/radianceCellScattering_Cubemap.ptx
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\bin2c.exe" -c --padd 0 --type char --name embedded_ptx_code_radiance_cell_scattering_cubemap mcrt-experiments/radianceCellScattering_Cubemap.ptx > mcrt-experiments/radianceCellScattering_Cubemap_embedded.c

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" -ptx mcrt-experiments/radianceCellScattering_Cubemap_hybrid_Octree.cu -I "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.5.0\include" -I "3rdparty/glm" -I "3rdparty/nvidia" --generate-line-info -use_fast_math -o mcrt-experiments/radianceCellScattering_Cubemap_hybrid_Octree.ptx
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\bin2c.exe" -c --padd 0 --type char --name embedded_ptx_code_radiance_cell_scattering_cubemap_octree mcrt-experiments/radianceCellScattering_Cubemap_hybrid_Octree.ptx > mcrt-experiments/radianceCellScattering_Cubemap_embedded_octree.c  

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" -ptx mcrt-experiments/unbiasedScattering.cu -I "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.5.0\include" -I "3rdparty/glm" -I "3rdparty/nvidia" --generate-line-info -use_fast_math -o mcrt-experiments/unbiasedScattering.ptx
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\bin2c.exe" -c --padd 0 --type char --name embedded_ptx_code_radiance_cell_scattering_unbiased mcrt-experiments/unbiasedScattering.ptx > mcrt-experiments/unbiasedScattering_embedded.c 

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" -ptx mcrt-experiments/unbiasedScatteringOctree.cu -I "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.5.0\include" -I "3rdparty/glm" -I "3rdparty/nvidia" --generate-line-info -use_fast_math -o mcrt-experiments/unbiasedScatteringOctree.ptx
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\bin2c.exe" -c --padd 0 --type char --name embedded_ptx_code_radiance_cell_scattering_unbiased_octree mcrt-experiments/unbiasedScatteringOctree.ptx > mcrt-experiments/unbiasedScattering_embedded_octree.c 

pause 