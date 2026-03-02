#version 430 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoords;
out vec2 TexCoords;

void main() {
  TexCoords = aTexCoords; // [0,1]
  gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0); // [-1, 1]
}