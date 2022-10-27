"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" -ptx mcrt-experiments/devicePrograms.cu -I "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.6.0\include" -I "3rdparty/glm" -I "3rdparty/nvidia" --generate-line-info -use_fast_math -o mcrt-experiments/devicePrograms.ptx
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\bin2c.exe" -c --padd 0 --type char --name embedded_ptx_code mcrt-experiments/devicePrograms.ptx > mcrt-experiments/devicePrograms_embedded.c 

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" -ptx mcrt-experiments/directLightingShaders.cu -I "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.6.0\include" -I "3rdparty/glm" -I "3rdparty/nvidia" --generate-line-info -use_fast_math -o mcrt-experiments/directLightingShaders.ptx
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\bin2c.exe" -c --padd 0 --type char --name embedded_ptx_code_direct_lighting mcrt-experiments/directLightingShaders.ptx > mcrt-experiments/directLightingShaders_embedded.c 

pause 