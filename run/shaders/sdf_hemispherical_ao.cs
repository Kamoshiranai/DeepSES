// Adapted from https://www.aduprat.com/portfolio/?page=articles/hemisphericalSDFAO

#version 430 core
layout(local_size_x = 4, local_size_y = 4) in;
layout(r32f, binding = 4) uniform image2D img_ao;

uniform sampler2D tex_pos; // in world space
uniform sampler2D tex_normal; // in world space
uniform sampler3D tex_sdf; // in model space

uniform vec3 dims; // size of sdf grid (512^3)
uniform float grid_res; // world size of an sdf grid voxel
uniform vec2 resolution; // screen size
uniform bool sdf_in_r_component; // whether sdf value is in .r component (else in .a component)

const float PI = 3.14159265359;

// From world space (molecule space) to model space (sdf grid space)
vec3 pos_in_grid(vec3 pos, vec3 dims) {
  return (((pos / grid_res) + (dims / 2)) / dims);
}

//Random number [0:1] without sine
#define HASHSCALE1 .1031
float hash(float p)
{
	vec3 p3  = fract(vec3(p) * HASHSCALE1);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

vec3 randomSphereDirExcludeNormal(vec2 rnd) //NOTE: project from uniform sample on cylinder to sphere but polar angle only in [pi/4,pi/2] (away from normal and -normal)
{
	float s = rnd.x*PI*2.;
	float t = rnd.y*2.-1.;
	return vec3(sin(s), cos(s), t) / sqrt(1.0 + t * t);
}
vec3 randomSphereDir(vec2 rnd) //NOTE: project from uniform sample on cylinder to sphere
{
	float s = rnd.x*PI*2.;
	float t = rnd.y*2.-1.;
    float radius = sqrt(1.0 - rnd.y*rnd.y);
	return vec3(sin(s)*radius, cos(s)*radius, t);
}
vec3 randomHemisphereDirExcludeNormal(vec3 dir, float i)
{
	vec3 v = randomSphereDirExcludeNormal( vec2(hash(i+1.), hash(i+2.)) );
	return v * sign(dot(v, dir));
}
vec3 randomHemisphereDir(vec3 dir, float i)
{
	vec3 v = randomSphereDir( vec2(hash(i+1.), hash(i+2.)) );
	return v * sign(dot(v, dir));
}

float hemisphericalAmbientOcclusion( vec3 pos, vec3 normal, float maxDist, float falloff )
{
	const int nbIte = 128;
    const float nbIteInv = 1./float(nbIte);
    // const float rad = 1.-1.*nbIteInv; //Hemispherical factor (self occlusion correction)
    
	float ao = 0.0;
    
    for( int i=0; i<nbIte; i++ )
    {
        float random_length = hash(float(i)) * maxDist;
        vec3 shifted_pos = pos + randomHemisphereDir(normal, random_length ) * random_length;
        // shifted_pos in model space
        vec3 shifted_pos_grid = pos_in_grid(shifted_pos, dims);
        // sdf value at shifted_pos
        float sdf_shifted_pos;
        if (sdf_in_r_component) {
            sdf_shifted_pos = texture(tex_sdf, shifted_pos_grid).r; //NOTE: for vdw use .a for deepses use .r
        } else {
            sdf_shifted_pos = texture(tex_sdf, shifted_pos_grid).a;
        }

        // ao += max(dot(rd,n) - max(map( p + rd ),0.), 0.) / maxDist * falloff;
        ao += abs(random_length - max(sdf_shifted_pos, 0.)) / maxDist * falloff;
	}
    float avg_ao = pow(ao * nbIteInv, 10);
    // return clamp( 1.-avg_ao, 0., 1.);
    return smoothstep(0., 1., 1.-avg_ao);
}


float classicAmbientOcclusion( vec3 pos, vec3 normal, float maxDist, float falloff )
{
	float ao = 0.0;
	const int nbIte = 128;
    for( int i=0; i<nbIte; i++ )
    {
        float random_length = hash(float(i)) * maxDist;
        vec3 shifted_pos = pos + normal*random_length;

        // shifted_pos in model space
        vec3 shifted_pos_grid = pos_in_grid(shifted_pos, dims);
        // sdf value at shifted_pos
        float sdf_shifted_pos = texture(tex_sdf, shifted_pos_grid).r;
        
        ao += (random_length - max(sdf_shifted_pos, 0.)) / maxDist * falloff;
    }
    float avg_ao = pow(ao/float(nbIte), 10);
    return clamp( 1.-avg_ao, 0., 1.);
}

void main() {

    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    vec2 normalized_pixel_coords = (1.0f * pixel_coords) / resolution;
    // position and normal in world space
    vec3 pos = texture(tex_pos, normalized_pixel_coords).rgb;
    vec3 normal = texture(tex_normal, normalized_pixel_coords).rgb;

    float ambient_occlusion;

    if (normal == vec3(0)) {
        // if not on molecule
        ambient_occlusion = 1.0;
    } else {
        // ao parameters
        // float maxDist = 100 * grid_res;
        float maxDist = 100 * grid_res * dims.x / 512; //NOTE: this helps to scale for different grid sizes than 512^3 //NOTE: /5 helps with molecules with more atoms
        float falloff = 2.25;

        ambient_occlusion = hemisphericalAmbientOcclusion(pos, normal, maxDist, falloff);
    }
    // save ao value in red component
    imageStore(img_ao, pixel_coords, vec4(ambient_occlusion, 0.0, 0.0, 1.0));
}