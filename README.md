# mcrt-draft
Render engine implementation for my Master Thesis subject: A Scalable and Coherent Approach to Monte Carlo Path Tracing. This thesis explores an adaptation of the Monte-Carlo path tracing algorithm that makes it more modular and scalable for cloud environments. The proposed adaptation computes diffuse irradiance in texture space. This introduces aliasing and texture seam artifacts, which are visible in the screenshot. Solutions for these artifacts exist, but since the focus of this work was to investigate the performance of the proposed algorithm, these mitigations are not implemented here.

# Setup and usage
* Make `models/` directory in root folder (contains world models that will be loaded)
* Make `data/world_data_textures/positions`, `data/world_data_textures/normals`,`data/world_data_textures/diffuse_coords` directories in root folder (contain world data that will be loaded, and the world data will be stored here)
* Change which models will be loaded in `App:loadScene()`
* In `app.h` you can change whether the biased or unbiased approach is used
* In `Renderer:initLightingTexturesPerObject` and `Renderer:prepareUVWorldPositionsPerObject`, uncomment the lines that correspond to your model (or add new ones in case you test with new scenes)
* Make sure you pass `LOAD_WORLD_DATA` or `CALCULATE_WORLD_DATA` when calling `Renderer:prepareUVWorldPositionsPerObject` depending on what you desire
* In `unbiasedScattering.cu` and `radianceCellScattering_Cubemap_hybrid.cu`, make sure you uncomment the correct line where `diffuseTexColor` is set
* You need one model with lightmap UV coordinates and a copy of the same model with diffuse UV coordinates!

# Screenshot
![Screenshot](./screenshot.png?raw=true "Sponza screenshot")
