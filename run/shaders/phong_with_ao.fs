#version 430 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D tex_pos;
uniform sampler2D tex_normal;
uniform sampler2D tex_color;
uniform sampler2D tex_ao;
uniform vec3 camera_pos;
uniform bool bool_uniform_color;
uniform vec3 uniform_color;

void main() {
  vec3 pos = texture(tex_pos, TexCoords).rgb;
  vec3 normal = texture(tex_normal, TexCoords).rgb;
  vec3 color = texture(tex_color, TexCoords).rgb;
  float ambient_occlusion = texture(tex_ao, TexCoords).r;

  if (normal == vec3(0))
    discard;

  vec3 light_position = camera_pos;
  vec3 view_dir = normalize(camera_pos - pos);
  vec3 light_dir = normalize(light_position - pos);
  vec3 reflect_dir = reflect(-light_dir, normal);

  float ambient = 0.7 * ambient_occlusion;
  float diffuse = 0.5 * max(0.0, dot(normal, light_dir));
  float specularStrength = 0.5;
  float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
  float specular = specularStrength * spec;

  if (bool_uniform_color)
    color = uniform_color;
  // vec3 col = (color * (ambient + diffuse + specular));
  // color = vec3(1.0);
  vec3 col = (color * (ambient + diffuse));

  // // add silhouette
  // if (abs(dot(normalize(camera_pos), normal)) <= 0.3) {
  //   col = vec3(0.0);
  // }

  FragColor = vec4(col, 1.0);
}