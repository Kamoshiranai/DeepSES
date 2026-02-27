
#version 430
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
layout(r32f, binding = 5) uniform image3D smoothed_b_factor_grid;
layout(binding = 3) uniform sampler3D b_factor_grid; 

void main() {
    // 1. Get exact integer voxel coordinates
    ivec3 pixel_coords = ivec3(gl_GlobalInvocationID.xyz);
    ivec3 dims = imageSize(smoothed_b_factor_grid);

    // 2. Prevent out-of-bounds execution if dispatch size isn't a perfect multiple of 8
    if (any(greaterThanEqual(pixel_coords, dims))) {
        return;
    }

    // 3. Helper function to clamp coordinates to the edges so we don't sample out-of-bounds 
    // (texelFetch returns 0.0 if out of bounds, which would incorrectly darken the edges)
    ivec3 max_bounds = dims - 1;

    float new_value = 0.0;
    
    // Center
    new_value += texelFetch(b_factor_grid, pixel_coords, 0).x;
    
    // X neighbors
    new_value += texelFetch(b_factor_grid, clamp(pixel_coords + ivec3(1, 0, 0), ivec3(0), max_bounds), 0).x;
    new_value += texelFetch(b_factor_grid, clamp(pixel_coords - ivec3(1, 0, 0), ivec3(0), max_bounds), 0).x;
    
    // Y neighbors
    new_value += texelFetch(b_factor_grid, clamp(pixel_coords + ivec3(0, 1, 0), ivec3(0), max_bounds), 0).x;
    new_value += texelFetch(b_factor_grid, clamp(pixel_coords - ivec3(0, 1, 0), ivec3(0), max_bounds), 0).x;
    
    // Z neighbors
    new_value += texelFetch(b_factor_grid, clamp(pixel_coords + ivec3(0, 0, 1), ivec3(0), max_bounds), 0).x;
    new_value += texelFetch(b_factor_grid, clamp(pixel_coords - ivec3(0, 0, 1), ivec3(0), max_bounds), 0).x;

    // Average
    new_value /= 7.0;

    // Write output
    imageStore(smoothed_b_factor_grid, pixel_coords, vec4(new_value, 0.0, 0.0, 0.0));
}