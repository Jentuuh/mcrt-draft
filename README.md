# mcrt-draft
Draft implementation for my Master Thesis subject: Memory Coherent Ray Tracing (MCRT).

# Setup and usage
* Make `models/` directory in root folder (contains world models that will be loaded)
* Make `data/world_data_textures/positions`, `data/world_data_textures/normals`,`data/world_data_textures/diffuse_coords` directories in root folder (contain world data that will be loaded, and the world data will be stored here)
* Change which models will be loaded in `App:loadScene()`
* In `app.h` you can change whether the biased or unbiased approach is used
* In `Renderer:initLightingTexturesPerObject` and `Renderer:prepareUVWorldPositionsPerObject`, uncomment the lines that correspond to your model (or add new ones in case you test with new scenes)
* Make sure you pass `LOAD_WORLD_DATA` or `CALCULATE_WORLD_DATA` when calling `Renderer:prepareUVWorldPositionsPerObject` depending on what you desire
* In `unbiasedScattering.cu` and `radianceCellScattering_Cubemap_hybrid.cu`, make sure you uncomment the correct line where `diffuseTexColor` is set
* You need one model with lightmap UV coordinates and a copy of the same model with diffuse UV coordinates!
