
#version 430
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
layout(r32f, binding = 0) uniform image3D img_dist_field;

layout(std430, binding = 1) buffer indexBuffer { uint index[]; };

uniform float resolution;
uniform float r_probe;
uniform int work_groups_per_call;
uniform int call;
float epsilon = 0.005;

// Get pos in atom space from coords in sdf grid space
vec3 coords(ivec3 pixel_coords, vec3 dims, float resolution) {
  return ((pixel_coords - dims / 2.0f) * resolution);
}

void main() {

  // Retrieve block index for refinement of a grid point in the boundary region
  ivec3 local_dims =
      ivec3(8, 8, 8); // This is the workgroupsize from the previous shader
  ivec3 block_dims = imageSize(img_dist_field) / local_dims;
  int block_ind = int(index[gl_WorkGroupID.x + call * work_groups_per_call]);

  // Get 3d coords from 1d index
  // First we calculate the coords for the first grid point in the index block
  ivec3 block_coords;
  block_coords.x = block_ind % block_dims.x;
  int rest = (block_ind - block_coords.x) / block_dims.x;
  block_coords.y = rest % block_dims.y;
  block_coords.z = (rest - block_coords.y) / block_dims.y;

  ivec3 pixel_coords = block_coords * local_dims;
  // Then we go to the specific grid point of this compute shader invocation
  pixel_coords += ivec3(gl_LocalInvocationID);

  vec4 result = imageLoad(img_dist_field, pixel_coords);
  // Atoms further than the cutoff do not need to be considered
  float cutoff = r_probe + resolution;
  int iter_max = int(cutoff / resolution);

  ivec3 global_dims = imageSize(img_dist_field);
  vec3 pos = coords(pixel_coords, global_dims, resolution);
  bool found = false;
  result.r = -resolution; //NOTE: now distance is stored in .r not .a!
  float min_dist = r_probe + resolution + epsilon;
  // Iterate over neighbouring grid points to see if we find a point outside of
  // the SES if there are several found, we use the one closest to our grid
  // point
  for (int x = -iter_max; x < iter_max; ++x) {
    for (int y = -iter_max; y < iter_max; ++y) {
      for (int z = -iter_max; z < iter_max; ++z) {
        ivec3 n_coords = pixel_coords + ivec3(x, y, z);
        vec3 n_pos = coords(n_coords, global_dims, resolution);
        if (distance(pos, n_pos) > min_dist) // only consider points closer than r_probe + size of grid voxel
          continue;
        vec4 n_result = imageLoad(img_dist_field, n_coords);
        if (abs(n_result.r - r_probe) < epsilon) {// check if point is outside relevant region (has sdf value larger than r_probe)
          min_dist = distance(pos, n_pos);
          found = true;
        }
      }
    }
  }
  if (found) {
    result.r = r_probe - min_dist;
  }

  imageStore(img_dist_field, pixel_coords, result);
}