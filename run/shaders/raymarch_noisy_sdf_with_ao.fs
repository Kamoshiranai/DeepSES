#version 430 core
// precision highp float;
// precision highp int;

in vec2 TexCoords; //NOTE: passthrough.vs returns values in [-1,1]
out vec4 fragColor;

layout(binding = 0) uniform sampler3D grid; // sdf values
layout(binding = 3) uniform sampler3D b_factor_grid; //NOTE: b-factor values are bound to 3, smoothed values to 5
layout(binding = 4) uniform sampler3D atom_color_grid;
const vec3 OBJECT_COLOR = vec3(0.79); // gamma corrected 0.9
uniform mat4 view;
uniform mat4 projection;
uniform vec3 camera_pos;
uniform vec3 camera_front;
uniform vec2 resolution;
uniform vec3 dims;
uniform vec3 dims_b_factor;
uniform float grid_res; // world space size (Angstrom) of one voxel
uniform float grid_res_b_factor; // world space size (Angstrom) of one voxel
uniform float screen_res; // world space size of one pixel, in A / px
uniform bool varyAmplitude; //NOTE: vary amplitude OR frequency

//NOTE: amplitude values are specified in a range of [0,1] which is reparametrized, s.t. the distances are perceptually uniform.
// We pass those values through a sigmoid (reparam.) and then through a linear transform mapping the values to the range [0.002, 0.14].
// Next we multiply by 250px to obtain the screen size. The corresponding world space size can be calculated by multiplying with the screen_res (A / px)
const float baseAmplitudePx = 250.0;
const float maxAmplitudeWorldUnits = 0.14 * baseAmplitudePx * screen_res;
//NOTE: frequency values are specified in a range of [0,1] which is reparametrized, s.t. the distances are perceptually uniform.
// We pass those values through a sigmoid (reparam.) and then through a linear transform mapping the values to the range [4, 25].
// Next we multiply by 1/250px to obtain the frequency in terms of screen size. The corresponding world space size can be calculated by dividing with the screen_res (A / px)
const float baseFrequencyPx = 1.0 / 250; 
const float defaultFrequencyWorldUnits = 16.0 * baseFrequencyPx / screen_res;
const float defaultAmplitudeWorldUnits = 0.048 * baseAmplitudePx * screen_res;
const bool binBFactor = true;
const int numBins = 5;
const float uncertainty = 1.0; // can be used to turn off noise

const float tilt = radians(93.2);
const float slant = radians(33.9);
const vec3 lightDir =
    vec3(cos((tilt)) * sin(slant), sin(tilt) * sin(slant), cos(slant));
// transform light dir from view space to world space
const vec3 lightDirViewSpace = (inverse(view) * vec4(lightDir, 0.0f)).xyz;
const float ASPECT = 1.0;
const float EPSILON = 0.001;
const float PI = 3.14159265359;
// const float MIN_STEP_SIZE = EPSILON * 0.1;
const float MIN_STEP_SIZE = grid_res / 100; //TODO making this smaller than grid_res / 10 leads to black transitions between areas of different b-factor, for diffuse shading, 
// but having it smaller gives better results for high freq. areas
const float SCALE_STEP_SIZE = 0.1; //0.1
const int MAX_STEPS = 5000; // 1500

float bFactorBinning(float b_factor) {
  float binWidth = 1.0 / numBins;
  return (floor(b_factor / binWidth) + 0.5) * binWidth; // divide [0,1] in equal-sized bins and map values to center of bin
}

float sigmoid(float a, float b, float x) {
  x = clamp(x, 0.0001, 0.9999);
  return 1.0 / (1.0 + (1.0/a - 1.0) * pow(1.0/x - 1.0, b));
}

// reparametrization of frequency
float perceptualFrequencyToWorldUnits(float freqPerceptual) {
  // invert mapping:
  // freqPerceptual = 1.0 - freqPerceptual;
  // b-factor binning
  if (binBFactor) freqPerceptual = bFactorBinning(freqPerceptual);
  // 1. perceptual frequencies are given in [0,1] and reparametrized using a fitted sigmoid
  // 2. map to range [4, 25]
  // 3. map to px 
  // 4. map to World units
  return sigmoid(0.2030, 0.9180, freqPerceptual)
  * (25. - 4.) + 4.
  * baseFrequencyPx 
  / screen_res; 
}

// reparametrization of amplitude
float perceptualAmplitudeToWorldUnits(float amplitudePerceptual) {
  // b-factor binning
  // if (binBFactor) amplitudePerceptual = bFactorBinning(amplitudePerceptual);
  // 1. perceptual amplitudes are given in [0,1] and reparametrized using a fitted sigmoid
  // 2. map to range [0.002, 0.14]
  // 3. map to px 
  // 4. map to World units
  return sigmoid(0.1441, 1.188, amplitudePerceptual)
  * (0.14 - 0.002) + 0.002
  * baseAmplitudePx 
  * screen_res;
}

// From global space (molecule space) to sdf grid space, normalized to [0,1]^3
vec3 pos_in_grid(vec3 pos, vec3 dimensions, float grid_resolution) {
  return (((pos / grid_resolution) + (dimensions / 2)) / dimensions);
}

vec3 getWorldPosfromScreenPos(vec2 screenPos) {
  vec4 worldPos =
      inverse(projection * view) * vec4(2.0f * screenPos - 1.0f, 0.0f, 1.0f);
  return worldPos.xyz / worldPos.w;
}

// Calculate normal via central differences (6 texture look ups)
vec3 calculateNormal(vec3 coords_grid, vec3 noiseGradient) {

  vec3 epsilon_vec = vec3(1.0f / dims.x, 0.0, 0.0);

  float gradient_x = textureLod(grid, coords_grid + epsilon_vec.xyy, 0.0).x -
                     textureLod(grid, coords_grid - epsilon_vec.xyy, 0.0).x;
  float gradient_y = textureLod(grid, coords_grid + epsilon_vec.yxy, 0.0).x -
                     textureLod(grid, coords_grid - epsilon_vec.yxy, 0.0).x;
  float gradient_z = textureLod(grid, coords_grid + epsilon_vec.yyx, 0.0).x -
                     textureLod(grid, coords_grid - epsilon_vec.yyx, 0.0).x;

  vec3 normal = normalize(vec3(gradient_x, gradient_y, gradient_z));

  return normalize(normal + noiseGradient);
}

float noisySceneSDF(vec3 pos, float uncertainty, out vec3 noiseGradient);

// ------------------------------------------------------------------
// #include "psrdnoise3.glsl"
// ------------------------------------------------------------------
//
// psrdnoise3.glsl
//
// Authors: Stefan Gustavson (stefan.gustavson@gmail.com)
// and Ian McEwan (ijm567@gmail.com)
// Version 2021-12-02, published under the MIT license (see below)
//
// Copyright (c) 2021 Stefan Gustavson and Ian McEwan.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//

//
// Periodic (tiling) 3-D simplex noise (tetrahedral lattice gradient noise)
// with rotating gradients and analytic derivatives.
//
// This is (yet) another variation on simplex noise. Unlike previous
// implementations, the grid is axis-aligned to permit rectangular tiling.
// The noise pattern can be made to tile seamlessly to any integer periods
// up to 289 units in the x, y and z directions. Specifying a longer
// period than 289 will result in errors in the noise field.
//
// This particular version of 3-D noise also implements animation by rotating
// the generating gradient at each lattice point around a pseudo-random axis.
// The rotating gradients give the appearance of a swirling motion, and
// can serve a similar purpose for animation as motion along the fourth
// dimension in 4-D noise. 
//
// The rotating gradients in conjunction with the built-in ability to
// compute exact analytic derivatives allow for "flow noise" effects
// as presented by Ken Perlin and Fabrice Neyret.
//

// Use Perlin's rotated grid instead of the new tiling grid?
// Enabling this adds about 1% to the execution time and
// requires all periods to be multiples of 3. Other
// integer periods can be specified, but when not evenly
// divisible by 3, the actual period will be 3 times longer.
// Take care not to overstep the maximum allowed period (288).
//#define PERLINGRID

// Enable faster gradient rotations?
// Enabling this saves about 10% on execution time,
// but the function will not run faster for alpha = 0.
//#define FASTROTATION


// Permutation polynomial for the hash value
vec4 permute(vec4 x) {
     vec4 xm = mod(x, 289.0);
     return mod(((xm*34.0)+10.0)*xm, 289.0);
}

//
// 3-D tiling simplex noise with rotating gradients and first order
// analytical derivatives.
// "vec3 x" is the point (x,y,z) to evaluate
// "vec3 period" is the desired periods along x,y,z, up to 289.
// (If Perlin's grid is used, multiples of 3 up to 288 are allowed.)
// "float alpha" is the rotation (in radians) for the swirling gradients.
// The "float" return value is the noise value, and
// the "out vec3 gradient" argument returns the x,y,z partial derivatives.
//
// The function executes 15-20% faster if alpha is constant == 0.0
// across all fragments being executed in parallel.
// (This speedup will not happen if FASTROTATION is enabled. Do not specify
// FASTROTATION if you are not actually going to use the rotation.)
//
// Setting any period to 0.0 or a negative value will skip the periodic
// wrap for that dimension. Setting all periods to 0.0 makes the function
// execute 10-15% faster.
//
// Not using the return value for the gradient will make the compiler
// eliminate the code for computing it. This speeds up the function by
// around 10%.
//
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


// ------------------------------------------------------------------
// #include "ambient_occlusion.glsl"
// ------------------------------------------------------------------
float uToFloat01(uint x) {
  return float(x) / 4294967296.0; // 2^32
}

// From http://jcgt.org/published/0009/03/02/
uvec3 pcg3d(uvec3 v) {
  v = v * 1664525u + 1013904223u;
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v ^= v >> 16u;
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  return v;
}
// cosine weighted hemisphere sampling
// rnd: vec2 of random numbers in [0,1)
// normal: vec3, the hemisphere axis
vec3 hemisphereSample(vec3 normal, int i) {
  uint x = uint(gl_FragCoord.x);
  uint y = uint(gl_FragCoord.y);
  uint z = uint(i);
  uvec3 seed = uvec3(x, y, z);
  uvec3 rndInt = pcg3d(seed);
  vec2 rnd = vec2(uToFloat01(rndInt.x), uToFloat01(rndInt.y));

  float phi = 2.0 * PI * rnd.x;       // azimuth
  float cosTheta = sqrt(1.0 - rnd.y); // polar, cosine-weighted
  float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

  // Cartesian coordinates in tangent space
  vec3 tangent, bitangent;
  if (abs(normal.x) > 0.1)
    tangent = normalize(cross(vec3(0, 1, 0), normal));
  else
    tangent = normalize(cross(vec3(1, 0, 0), normal));
  bitangent = cross(normal, tangent);

  return normalize(tangent * (cos(phi) * sinTheta) +
                   bitangent * (sin(phi) * sinTheta) + normal * cosTheta);
}

float rayTracedAO(vec3 pos, vec3 normal) {
  float ao = 0.0;
  int nSamples = 200;
  vec3 noiseGradient; // needs to be given to noise computation even though we
                      // don't need it here
  float uncertainty = 1.0; 
  float maxDist = grid_res * dims.x / 4;
  int maxSteps = 100;
  for (int i = 0; i < nSamples; i++) {
    vec3 dir = hemisphereSample(normal, i);
    // float t = 0.; //NOTE: would this not mean we get an immediate hit at the starting position? but not if we take dist * 0.5 in raymarching and dist in ao ...
    //NOTE: what happens here can be "fixed" by either using a smaller MIN_STEP_SIZE or using a positive t at the start.
    float t = grid_res * dims.x / 100;
    bool hit = false;
    for (int j = 0; j < maxSteps; j++) {
      vec3 shiftedPos = pos + dir * t;
      float dist = noisySceneSDF(shiftedPos, uncertainty, noiseGradient);
      if (dist < EPSILON) {
        hit = true;
        break;
      }
      // dist *= SCALE_STEP_SIZE; //TODO looks similar without this and is faster
      dist = max(dist, MIN_STEP_SIZE);
      t += dist;
      if (t > maxDist)
        break;
    }
    ao += hit ? 0. : 1.;
  }
  ao /= float(nSamples);
  return ao;
}
// ------------------------------------------------------------------------------------

// Add noise to distance value
float addNoise(vec3 pos, float frequency, float amplitude, float uncertainty, out vec3 gradient) {
  float noise;
  noise =
      psrdnoise(pos * frequency, vec3(0), 0.0, gradient);
  gradient *=
      amplitude * uncertainty *
      frequency; // NOTE: need to multiply by frequency (inner derivative)

  return noise * amplitude * uncertainty;
}

float noisySceneSDF(vec3 pos, float uncertainty, out vec3 noiseGradient) {
    // modify amplitude / frequency accordint to B-factor
    vec3 coords_grid_b_factor = pos_in_grid(pos, dims_b_factor, grid_res_b_factor);
    float b_factor = textureLod(b_factor_grid, coords_grid_b_factor, 0.0).x;
    float localAmp = varyAmplitude ? perceptualAmplitudeToWorldUnits(b_factor) : (defaultAmplitudeWorldUnits);
    float localFreq = varyAmplitude ? (defaultFrequencyWorldUnits) : perceptualFrequencyToWorldUnits(b_factor);
    vec3 coords_grid = pos_in_grid(pos, dims, grid_res);
    float dist = textureLod(grid, coords_grid, 0.0).x;
    dist += addNoise(pos, localFreq, localAmp, uncertainty, noiseGradient); 
    return dist;
}

float calculateDiffuse(vec3 pos, vec3 normal, vec3 eyePos) {
  vec3 viewDir = normalize(eyePos - pos);
  vec3 reflectDir = reflect(-lightDirViewSpace, normal);

  float diffuseFactor = max(dot(normal, lightDirViewSpace), 0.0);

  return diffuseFactor;
}

void main() {

  // vec3 rayOrigin = vec3(v_uv * vec2(ASPECT, 1.0), CAMERA_DISTANCE); //NOTE: this also would work for orthographic proj.
  vec3 rayOrigin = getWorldPosfromScreenPos(TexCoords);
      // 1.0f * pixel_coords /
      // resolution); // multiplication with 1.0f necessary, otherwise int division
  vec3 rayDirection = normalize(camera_front);

  // --- Raymarching loop ---
  float rayDepth = 0.0;
  float dist;
  vec3 pos;
  bool hit = false;
  vec3 noiseGradient;

  for (int i = 0; i < MAX_STEPS; i++) {
    pos = rayOrigin + rayDepth * rayDirection;
    dist = noisySceneSDF(pos, uncertainty, noiseGradient);
    // dist = noisySceneSDF(pos, uncertainty, noiseGradient) * 0.5; //TODO
    if (dist < EPSILON) {
      hit = true;
      break;
    }
    if (dist < EPSILON + 1.5 * maxAmplitudeWorldUnits) { 
      dist *= SCALE_STEP_SIZE;
    } else {
      dist *= 0.5;
    }
    dist = max(dist, MIN_STEP_SIZE);
    rayDepth += dist;

    if (rayDepth > grid_res * dims.x * 5)
      break;
  }

  // vec3 color = vec3(0.217); // gamma corrected 0.5
  vec3 color = vec3(0.029); // gamma corrected 0.2
  float ao = 0.0;
  float diffuse = 0.0;

  if (hit) {
    vec3 coords_grid = pos_in_grid(pos, dims, grid_res);
    // vec3 coords_grid_b_factor = pos_in_grid(pos, dims_b_factor, grid_res_b_factor);
    // float b_factor = textureLod(b_factor_grid, coords_grid_b_factor, 0.0).x;
    vec3 atom_color = textureLod(atom_color_grid, coords_grid, 0.0).xyz;
    vec3 normal = calculateNormal(coords_grid, noiseGradient);
    diffuse = calculateDiffuse(pos, normal, rayOrigin);
    ao = rayTracedAO(pos, normal);
    color = diffuse * 0.8 * OBJECT_COLOR + 0.2 * ao * OBJECT_COLOR; 
    // color = diffuse * 0.8 * atom_color + 0.2 * ao * atom_color; 
    // color = vec3(b_factor);
  }
  // gamma
  color = pow(color, vec3(0.4545));

  fragColor = vec4(color, 1.0);
}