"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" -ptx mcrt-experiments/textureTesting.cu -I "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.6.0\include" -I "3rdparty/glm" -I "3rdparty/nvidia" --generate-line-info -use_fast_math -o mcrt-experiments/radianceCellScattering_Cubemap.ptx
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\bin2c.exe" -c --padd 0 --type char --name embedded_ptx_code_radiance_cell_scattering_cubemap mcrt-experiments/radianceCellScattering_Cubemap.ptx > mcrt-experiments/radianceCellScattering_Cubemap_embedded.c

pause