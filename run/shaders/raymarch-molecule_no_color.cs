
#version 430
layout(local_size_x = 4, local_size_y = 4) in;
layout(rgba32f, binding = 1) uniform image2D img_pos;
layout(rgba32f, binding = 2) uniform image2D img_normal;
layout(rgba32f, binding = 3) uniform image2D img_color;

layout(binding = 0) uniform sampler3D grid;

uniform mat4 view;
uniform mat4 projection;
uniform vec3 camera_pos;
uniform vec3 camera_front;
uniform vec2 resolution;
uniform float epsilon = 0.001;
uniform vec3 dims;
uniform float grid_res;

// From global space (molecule space) to sdf grid space
vec3 pos_in_grid(vec3 pos, vec3 dims) {
  return (((pos / grid_res) + (dims / 2)) / dims);
}

vec3 getWorldPosfromScreenPos(vec2 screenPos) {
  vec4 worldPos =
      inverse(projection * view) * vec4(2.0f * screenPos - 1.0f, 0.0f, 1.0f);
  return worldPos.xyz / worldPos.w;
}

// Calculate normal via central differences (6 texture look ups)
vec3 calculate_normal(vec3 coords_grid) {

  vec3 epsilon_vec = vec3(1.0f / dims.x, 0.0, 0.0);

  float gradient_x = texture(grid, coords_grid + epsilon_vec.xyy).x -
                     texture(grid, coords_grid - epsilon_vec.xyy).x;
  float gradient_y = texture(grid, coords_grid + epsilon_vec.yxy).x -
                     texture(grid, coords_grid - epsilon_vec.yxy).x;
  float gradient_z = texture(grid, coords_grid + epsilon_vec.yyx).x -
                     texture(grid, coords_grid - epsilon_vec.yyx).x;

  vec3 normal = vec3(gradient_x, gradient_y, gradient_z);

  return normalize(normal);
}

// Calculate normal via 3D Sobel (27 texture look ups)
// Implementation according to Wikipedia
// https://en.wikipedia.org/wiki/Sobel_operator#Extension_to_other_dimensions
vec3 sobel_normal(vec3 coords_grid) {
  vec3 h = vec3(1, 2, 1);
  vec3 d = vec3(1, 0, -1);          // h' in Wikipedia
  vec3 e = vec3(-1, 0, 1) / dims.x; // Access to the neighboring grid points

  float xxx = texture(grid, coords_grid + e.xxx).x;
  float xxy = texture(grid, coords_grid + e.xxy).x;
  float xxz = texture(grid, coords_grid + e.xxz).x;
  float xyx = texture(grid, coords_grid + e.xyx).x;
  float xzx = texture(grid, coords_grid + e.xzx).x;
  float xyy = texture(grid, coords_grid + e.xyy).x;
  float xyz = texture(grid, coords_grid + e.xyz).x;
  float xzy = texture(grid, coords_grid + e.xzy).x;
  float xzz = texture(grid, coords_grid + e.xzz).x;

  float yxx = texture(grid, coords_grid + e.yxx).x;
  float yxy = texture(grid, coords_grid + e.yxy).x;
  float yxz = texture(grid, coords_grid + e.yxz).x;
  float yyx = texture(grid, coords_grid + e.yyx).x;
  float yyz = texture(grid, coords_grid + e.yyz).x;
  float yyy = texture(grid, coords_grid + e.yyy).x;
  float yzx = texture(grid, coords_grid + e.yzx).x;
  float yzy = texture(grid, coords_grid + e.yzy).x;
  float yzz = texture(grid, coords_grid + e.yzz).x;

  float zxx = texture(grid, coords_grid + e.zxx).x;
  float zxy = texture(grid, coords_grid + e.zxy).x;
  float zxz = texture(grid, coords_grid + e.zxz).x;
  float zyx = texture(grid, coords_grid + e.zyx).x;
  float zyy = texture(grid, coords_grid + e.zyy).x;
  float zyz = texture(grid, coords_grid + e.zyz).x;
  float zzx = texture(grid, coords_grid + e.zzx).x;
  float zzy = texture(grid, coords_grid + e.zzy).x;
  float zzz = texture(grid, coords_grid + e.zzz).x;

  float gradient_x =
      xxx * d.x * h.x * h.x + xxy * d.x * h.x * h.y + xxz * d.x * h.x * h.z +
      xyx * d.x * h.y * h.x + xzx * d.x * h.z * h.x + xyy * d.x * h.y * h.y +
      xyz * d.x * h.y * h.z + xzy * d.x * h.z * h.y + xzz * d.x * h.z * h.z +
      yxx * d.y * h.x * h.x + yxy * d.y * h.x * h.y + yxz * d.y * h.x * h.z +
      yyx * d.y * h.y * h.x + yyz * d.y * h.y * h.z + yyy * d.y * h.y * h.y +
      yzx * d.y * h.z * h.x + yzy * d.y * h.z * h.y + yzz * d.y * h.z * h.z +
      zxx * d.z * h.x * h.x + zxy * d.z * h.x * h.y + zxz * d.z * h.x * h.z +
      zyx * d.z * h.y * h.x + zyy * d.z * h.y * h.y + zyz * d.z * h.y * h.z +
      zzx * d.z * h.z * h.x + zzy * d.z * h.z * h.y + zzz * d.z * h.z * h.z;
  float gradient_y =
      xxx * h.x * d.x * h.x + xxy * h.x * d.x * h.y + xxz * h.x * d.x * h.z +
      xyx * h.x * d.y * h.x + xzx * h.x * d.z * h.x + xyy * h.x * d.y * h.y +
      xyz * h.x * d.y * h.z + xzy * h.x * d.z * h.y + xzz * h.x * d.z * h.z +
      yxx * h.y * d.x * h.x + yxy * h.y * d.x * h.y + yxz * h.y * d.x * h.z +
      yyx * h.y * d.y * h.x + yyz * h.y * d.y * h.z + yyy * h.y * d.y * h.y +
      yzx * h.y * d.z * h.x + yzy * h.y * d.z * h.y + yzz * h.y * d.z * h.z +
      zxx * h.z * d.x * h.x + zxy * h.z * d.x * h.y + zxz * h.z * d.x * h.z +
      zyx * h.z * d.y * h.x + zyy * h.z * d.y * h.y + zyz * h.z * d.y * h.z +
      zzx * h.z * d.z * h.x + zzy * h.z * d.z * h.y + zzz * h.z * d.z * h.z;
  float gradient_z =
      xxx * h.x * h.x * d.x + xxy * h.x * h.x * d.y + xxz * h.x * h.x * d.z +
      xyx * h.x * h.y * d.x + xzx * h.x * h.z * d.x + xyy * h.x * h.y * d.y +
      xyz * h.x * h.y * d.z + xzy * h.x * h.z * d.y + xzz * h.x * h.z * d.z +
      yxx * h.y * h.x * d.x + yxy * h.y * h.x * d.y + yxz * h.y * h.x * d.z +
      yyx * h.y * h.y * d.x + yyz * h.y * h.y * d.z + yyy * h.y * h.y * d.y +
      yzx * h.y * h.z * d.x + yzy * h.y * h.z * d.y + yzz * h.y * h.z * d.z +
      zxx * h.z * h.x * d.x + zxy * h.z * h.x * d.y + zxz * h.z * h.x * d.z +
      zyx * h.z * h.y * d.x + zyy * h.z * h.y * d.y + zyz * h.z * h.y * d.z +
      zzx * h.z * h.z * d.x + zzy * h.z * h.z * d.y + zzz * h.z * h.z * d.z;

  return -1 * normalize(vec3(gradient_x, gradient_y, gradient_z));
}

void main() {

  ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
  vec3 rayOrigin = getWorldPosfromScreenPos(
      1.0f * pixel_coords /
      resolution); // multiplication with 1.0f necessary, otherwise int division
  vec3 rayDirection = normalize(camera_front);

  // raymarch
  vec3 pos, normal, color, coords_grid;
  float depth = 0.0;
  int max_steps = 2000;
  // bool was_inside_grid = false;
  // bool inside_grid;
  float dist;
  int i;
  // float last_dist;
  for (i = 0; i < max_steps; i++) {
    pos = rayOrigin + depth * rayDirection;
    coords_grid = pos_in_grid(pos, dims);
    dist = texture(grid, coords_grid).x;

    if (dist < epsilon)
      break;
    depth += max(dist, grid_res / 10);
  }
  if (i < max_steps) { //} && inside_grid) {
    normal = calculate_normal(coords_grid);
  } else {
    normal = vec3(0.0);
  }

  color = vec3(0.0); // just dummy color

  imageStore(img_pos, pixel_coords, vec4(pos, 1.0));
  imageStore(img_normal, pixel_coords, vec4(normal, 1.0));
  imageStore(img_color, pixel_coords, vec4(color, 1.0));
}