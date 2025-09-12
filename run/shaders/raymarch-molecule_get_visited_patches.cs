#version 430
layout(local_size_x = 4, local_size_y = 4) in;

// Input Sampler
layout(binding = 0) uniform sampler3D grid; // Distance field

// Uniforms 
uniform mat4 view;
uniform mat4 projection;
uniform vec3 camera_pos; //NOTE: camera pos in world space ?
uniform vec3 camera_front; //NOTE: camera direction in world space
uniform vec2 resolution;
uniform float epsilon = 0.001;
uniform vec3 dims; // Total grid dimensions in voxels (e.g., 512, 512, 512)
uniform float grid_res; // Size of one voxel (in 512, 512, 512 grid) in Angstrom (world space units)

// *** NEW Uniforms ***
uniform vec3 patch_dims_voxels; // Size of one patch in voxels (e.g., 64, 64, 64)
uniform ivec3 patches_per_dim;  // Number of patches along each axis (e.g., 8, 8, 8)

// *** NEW SSBO for Patch Visibility ***
layout(std430, binding = 4) buffer PatchVisibility {
    int visited_patches[]; // Size 512 (8*8*8), initialized to 0 on CPU/GPU
};

//DEBUG:
layout(std430, binding = 5) buffer debugSSBO {
    float debug_values[]; // Size 512, init to zero on GPU
};


// --- Helper Functions ---
vec3 pos_in_grid(vec3 world_pos, vec3 grid_dims_voxels) { //NOTE: maps from world space to coords in grid normalized to [0,1]
  vec3 grid_origin_offset = grid_dims_voxels * grid_res * 0.5; // places the center of the grid in the origin of the world space (has units in Angstrom!)
  vec3 pos_relative_to_origin = world_pos + grid_origin_offset; // coordinates relative to grid origin in units of world space
  vec3 voxel_coords_float = pos_relative_to_origin / grid_res;
  return voxel_coords_float / grid_dims_voxels; // divide by size of grid in Angstrom to normalize to [0, 1] //NOTE: we could land outside of [0,1]
}

vec3 getWorldPosfromScreenPos(vec2 screenPos) {
  vec4 worldPos =
      inverse(projection * view) * vec4(2.0f * screenPos - 1.0f, 0.0f, 1.0f); //NOTE: view: world space -> camera/view space, projection: view space -> clip space //NOTE: scale and shift gets us Normalized Device Coordinates (NDC) in  [-1,1] //NOTE so this does clip space to world space but using screen coords / NDC instead of clip space coords...?
  return worldPos.xyz / worldPos.w; //NOTE: perspective divide

// --- Main Function ---
void main() {

  ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
  if (pixel_coords.x >= int(resolution.x) || pixel_coords.y >= int(resolution.y)) { // check if pixel in screen
      return;
  }

  // //DEBUG:
  // if ( (pixel_coords.x < 512) && ( (pixel_coords.x == 19) || (pixel_coords.x == 20) || (pixel_coords.x == 83) || (pixel_coords.x == 84) || (pixel_coords.x == 99) || (pixel_coords.x == 147) || (pixel_coords.x == 148) || (pixel_coords.x == 163) || (pixel_coords.x == 211) || (pixel_coords.x == 212) || (pixel_coords.x == 213) ) ) {
  //   visited_patches[pixel_coords.x] = 0;
  // }
  // else if (pixel_coords.x < 512) {
  //   atomicMax(visited_patches[pixel_coords.x], 1);
  // }
  // return;

  // --- Ray Setup (Perspective Projection) ---
  //  vec2 screenPosNDC = (vec2(pixel_coords) + vec2(0.5)) / resolution * 2.0 - 1.0;
  //  vec4 viewPos = inverse(projection) * vec4(screenPosNDC, -1.0, 1.0);
  //  viewPos /= viewPos.w;
  //  vec4 worldPosNear = inverse(view) * vec4(viewPos.xyz, 1.0);
  //  vec3 rayOrigin = camera_pos;
  //  vec3 rayDirection = normalize(worldPosNear.xyz - camera_pos);
  vec3 rayOrigin = getWorldPosfromScreenPos(
      1.0f * pixel_coords /
      resolution); // multiplication with 1.0f necessary, otherwise int division //NOTE: Screen pos in [0,1] x [0,1]
  vec3 rayDirection = normalize(camera_front);

  // --- Raymarch ---
  vec3 pos, coords_grid;
  float depth = 0.0;
  int max_steps = 1000;
  float dist = 0.0;
  int i;
  bool hit = false;
  int last_marked_patch_index = -1; // Track last marked patch to avoid redundant atomics

  for (i = 0; i < max_steps; i++) {
    pos = rayOrigin + depth * rayDirection; // in world space
    coords_grid = pos_in_grid(pos, dims); // normalized to [0,1], but can be outside of grid

    //DEBUG:
    // if ( (i == 1) && (pixel_coords.x == 256) && (pixel_coords.y < 512) ) {
    //   debug_values[pixel_coords.y] = pos.y;
    // }

    // // Check if the coordinate was inside and now is outside the grid boundaries [0, 1]
    // if ( (last_marked_patch_index != -1)  && ( any( lessThan(coords_grid, vec3(0.0)) ) || any( greaterThan(coords_grid, vec3(1.0)) ) ) ) {
    //   //DEBUG:
    //   if ( (pixel_coords.x == 256) && (pixel_coords.y < 512) ) {
    //     // debug_values[pixel_coords.y] = coords_grid.z; //NOTE: strange!
    //     debug_values[pixel_coords.y] = last_marked_patch_index * 1.0f;
    //   }
    //   break; // Ray exited the grid volume
    // }

    //NOTE: skip writing to buffer if we are not inside the grid
    if ( all( greaterThanEqual(coords_grid, vec3(0.0)) ) && all( lessThanEqual(coords_grid, vec3(1.0)) ) ) {
  
      // *** MARK VISITED PATCH (Every Step Inside Grid) ***
      // Calculate voxel coordinates (non-normalized)
      vec3 voxel_coord = coords_grid * dims;

      // Calculate 3D patch index (integer division effectively floors)
      // Clamp coordinates slightly inwards to avoid boundary issues with floor/int conversion
      vec3 safe_voxel_coord = clamp(voxel_coord, vec3(0.0), dims - vec3(1.0));
      ivec3 patch_coord_3d = ivec3(floor(safe_voxel_coord / patch_dims_voxels));

      // Clamp patch coordinates just in case (robustness)
      patch_coord_3d = clamp(patch_coord_3d, ivec3(0), patches_per_dim - ivec3(1));

      // Calculate linear patch index (e.g., X + Y*SizeX + Z*SizeX*SizeY)
      // Assuming Px=patches_per_dim.x, Py=patches_per_dim.y, Pz=patches_per_dim.z
      int patch_index_linear = patch_coord_3d.x +
                              patch_coord_3d.y * patches_per_dim.x +
                              patch_coord_3d.z * patches_per_dim.x * patches_per_dim.y;

      // Optimization: Only perform atomic if we entered a *new* patch
      // and check bounds
      if (patch_index_linear != last_marked_patch_index &&
          patch_index_linear >= 0 && patch_index_linear < 512)
      {
          atomicMax(visited_patches[patch_index_linear], 1);
          last_marked_patch_index = patch_index_linear; // Update last marked patch
          // //DEBUG:
          // if ( (pixel_coords.x == 256) && (pixel_coords.y < 512) ) {
          //   debug_values[pixel_coords.y] = float(patch_index_linear);
          // }
      }
      // ************************************************
    } else if (last_marked_patch_index != -1) {
      // NOTE we were inside and now we are outside the grid
      break;
    }

    // Sample distance field
    dist = texture(grid, coords_grid).x;

    // Check for hit (surface found)
    if (dist < epsilon) {
        // hit = true;
        // //DEBUG:
        // debug_values[last_marked_patch_index] = 1.0;
        break;
    }

    // Advance ray
    // Note: If 'dist' is guaranteed small enough to not skip patches,
    // just 'depth += dist' is fine. Using max adds robustness.
    // depth += max(epsilon * 1.1, dist);
    depth += dist;
  }
  
}