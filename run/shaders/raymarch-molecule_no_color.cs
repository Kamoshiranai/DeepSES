
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
// Permutation polynomial for the hash value
vec4 permute(vec4 x) {
     vec4 xm = mod(x, 289.0);
     return mod(((xm*34.0)+10.0)*xm, 289.0);
}

float psrdnoise(vec3 x, vec3 period, float alpha, out vec3 gradient)
{

#ifndef PERLINGRID
  // Transformation matrices for the axis-aligned simplex grid
  const mat3 M = mat3(0.0, 1.0, 1.0,
                      1.0, 0.0, 1.0,
                      1.0, 1.0, 0.0);

  const mat3 Mi = mat3(-0.5, 0.5, 0.5,
                        0.5,-0.5, 0.5,
                        0.5, 0.5,-0.5);
#endif

  vec3 uvw;

  // Transform to simplex space (tetrahedral grid)
#ifndef PERLINGRID
  // Use matrix multiplication, let the compiler optimise
  uvw = M * x;
#else
  // Optimised transformation to uvw (slightly faster than
  // the equivalent matrix multiplication on most platforms)
  uvw = x + dot(x, vec3(1.0/3.0));
#endif

  // Determine which simplex we're in, i0 is the "base corner"
  vec3 i0 = floor(uvw);
  vec3 f0 = fract(uvw); // coords within "skewed cube"

  // To determine which simplex corners are closest, rank order the
  // magnitudes of u,v,w, resolving ties in priority order u,v,w,
  // and traverse the four corners from largest to smallest magnitude.
  // o1, o2 are offsets in simplex space to the 2nd and 3rd corners.
  vec3 g_ = step(f0.xyx, f0.yzz); // Makes comparison "less-than"
  vec3 l_ = 1.0 - g_;             // complement is "greater-or-equal"
  vec3 g = vec3(l_.z, g_.xy);
  vec3 l = vec3(l_.xy, g_.z);
  vec3 o1 = min( g, l );
  vec3 o2 = max( g, l );

  // Enumerate the remaining simplex corners
  vec3 i1 = i0 + o1;
  vec3 i2 = i0 + o2;
  vec3 i3 = i0 + vec3(1.0);

  vec3 v0, v1, v2, v3;

  // Transform the corners back to texture space
#ifndef PERLINGRID
  v0 = Mi * i0;
  v1 = Mi * i1;
  v2 = Mi * i2;
  v3 = Mi * i3;
#else
  // Optimised transformation (mostly slightly faster than a matrix)
  v0 = i0 - dot(i0, vec3(1.0/6.0));
  v1 = i1 - dot(i1, vec3(1.0/6.0));
  v2 = i2 - dot(i2, vec3(1.0/6.0));
  v3 = i3 - dot(i3, vec3(1.0/6.0));
#endif

  // Compute vectors to each of the simplex corners
  vec3 x0 = x - v0;
  vec3 x1 = x - v1;
  vec3 x2 = x - v2;
  vec3 x3 = x - v3;

  if(any(greaterThan(period, vec3(0.0)))) {
    // Wrap to periods and transform back to simplex space
    vec4 vx = vec4(v0.x, v1.x, v2.x, v3.x);
    vec4 vy = vec4(v0.y, v1.y, v2.y, v3.y);
    vec4 vz = vec4(v0.z, v1.z, v2.z, v3.z);
	// Wrap to periods where specified
	if(period.x > 0.0) vx = mod(vx, period.x);
	if(period.y > 0.0) vy = mod(vy, period.y);
	if(period.z > 0.0) vz = mod(vz, period.z);
    // Transform back
#ifndef PERLINGRID
    i0 = M * vec3(vx.x, vy.x, vz.x);
    i1 = M * vec3(vx.y, vy.y, vz.y);
    i2 = M * vec3(vx.z, vy.z, vz.z);
    i3 = M * vec3(vx.w, vy.w, vz.w);
#else
    v0 = vec3(vx.x, vy.x, vz.x);
    v1 = vec3(vx.y, vy.y, vz.y);
    v2 = vec3(vx.z, vy.z, vz.z);
    v3 = vec3(vx.w, vy.w, vz.w);
    // Transform wrapped coordinates back to uvw
    i0 = v0 + dot(v0, vec3(1.0/3.0));
    i1 = v1 + dot(v1, vec3(1.0/3.0));
    i2 = v2 + dot(v2, vec3(1.0/3.0));
    i3 = v3 + dot(v3, vec3(1.0/3.0));
#endif
	// Fix rounding errors
    i0 = floor(i0 + 0.5);
    i1 = floor(i1 + 0.5);
    i2 = floor(i2 + 0.5);
    i3 = floor(i3 + 0.5);
  }

  // Compute one pseudo-random hash value for each corner
  vec4 hash = permute( permute( permute( 
              vec4(i0.z, i1.z, i2.z, i3.z ))
            + vec4(i0.y, i1.y, i2.y, i3.y ))
            + vec4(i0.x, i1.x, i2.x, i3.x ));

  // Compute generating gradients from a Fibonacci spiral on the unit sphere
  vec4 theta = hash * 3.883222077;  // 2*pi/golden ratio
  vec4 sz    = hash * -0.006920415 + 0.996539792; // 1-(hash+0.5)*2/289
  vec4 psi   = hash * 0.108705628 ; // 10*pi/289, chosen to avoid correlation

  vec4 Ct = cos(theta);
  vec4 St = sin(theta);
  vec4 sz_prime = sqrt( 1.0 - sz*sz ); // s is a point on a unit fib-sphere

  vec4 gx, gy, gz;

  // Rotate gradients by angle alpha around a pseudo-random ortogonal axis
#ifdef FASTROTATION
  // Fast algorithm, but without dynamic shortcut for alpha = 0
  vec4 qx = St;         // q' = norm ( cross(s, n) )  on the equator
  vec4 qy = -Ct; 
  vec4 qz = vec4(0.0);

  vec4 px =  sz * qy;   // p' = cross(q, s)
  vec4 py = -sz * qx;
  vec4 pz = sz_prime;

  psi += alpha;         // psi and alpha in the same plane
  vec4 Sa = sin(psi);
  vec4 Ca = cos(psi);

  gx = Ca * px + Sa * qx;
  gy = Ca * py + Sa * qy;
  gz = Ca * pz + Sa * qz;
#else
  // Slightly slower algorithm, but with g = s for alpha = 0, and a
  // useful conditional speedup for alpha = 0 across all fragments
  if(alpha != 0.0) {
    vec4 Sp = sin(psi);          // q' from psi on equator
    vec4 Cp = cos(psi);

    vec4 px = Ct * sz_prime;     // px = sx
    vec4 py = St * sz_prime;     // py = sy
    vec4 pz = sz;

    vec4 Ctp = St*Sp - Ct*Cp;    // q = (rotate( cross(s,n), dot(s,n))(q')
    vec4 qx = mix( Ctp*St, Sp, sz);
    vec4 qy = mix(-Ctp*Ct, Cp, sz);
    vec4 qz = -(py*Cp + px*Sp);

    vec4 Sa = vec4(sin(alpha));       // psi and alpha in different planes
    vec4 Ca = vec4(cos(alpha));

    gx = Ca * px + Sa * qx;
    gy = Ca * py + Sa * qy;
    gz = Ca * pz + Sa * qz;
  }
  else {
    gx = Ct * sz_prime;  // alpha = 0, use s directly as gradient
    gy = St * sz_prime;
    gz = sz;  
  }
#endif

  // Reorganize for dot products below
  vec3 g0 = vec3(gx.x, gy.x, gz.x);
  vec3 g1 = vec3(gx.y, gy.y, gz.y);
  vec3 g2 = vec3(gx.z, gy.z, gz.z);
  vec3 g3 = vec3(gx.w, gy.w, gz.w);

  // Radial decay with distance from each simplex corner
  vec4 w = 0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3));
  w = max(w, 0.0);
  vec4 w2 = w * w;
  vec4 w3 = w2 * w;

  // The value of the linear ramp from each of the corners
  vec4 gdotx = vec4(dot(g0,x0), dot(g1,x1), dot(g2,x2), dot(g3,x3));

  // Multiply by the radial decay and sum up the noise value
  float n = dot(w3, gdotx);

  // Compute the first order partial derivatives
  vec4 dw = -6.0 * w2 * gdotx;
  vec3 dn0 = w3.x * g0 + dw.x * x0;
  vec3 dn1 = w3.y * g1 + dw.y * x1;
  vec3 dn2 = w3.z * g2 + dw.z * x2;
  vec3 dn3 = w3.w * g3 + dw.w * x3;
  gradient = 39.5 * (dn0 + dn1 + dn2 + dn3);

  // Scale the return value to fit nicely into the range [-1,1]
  return 39.5 * n;
}



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

// Add perlin noise to distance value
float add_noise(vec3 pos) {
  float scale_amplitude = 1.0f;
  float frequency = 2.0f;
  vec3 gradient;
  float modulate_dist =
      scale_amplitude * psrdnoise(pos * frequency , vec3(0), 0.0, gradient);

  return modulate_dist;
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
  int max_steps = 1000;
  // bool was_inside_grid = false;
  // bool inside_grid;
  float dist;
  int i;
  // float last_dist;
  for (i = 0; i < max_steps; i++) {
    pos = rayOrigin + depth * rayDirection;
    coords_grid = pos_in_grid(pos, dims);
    dist = texture(grid, coords_grid).x;
    dist += add_noise(pos);

    if (dist < epsilon)
      break;
    depth += 0.1 * max(dist, grid_res / 10);
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