#ifndef UTILITIES_H
#define UTILITIES_H

#include <glad/glad.h>

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stb_image/stb_image.h>
#include <stb_image/stb_image_write.h>
#include <string>
#include <vector>

void screenshot(int width, int height) {
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
  auto str = oss.str();

  GLubyte *screen = new GLubyte[width * height * 4];
  stbi_flip_vertically_on_write(true);
  glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, screen);
  //   std::string name = "../../pics/screenshots/screenshot_" + str + ".png";
  std::string name = "../../gfx/screenshots/screenshot_" + str + ".png";
  const char *c_name = name.c_str();

  int write_png = stbi_write_png(c_name, width, height, 4, screen, 0);
  if (write_png != 0) {
    std::cout << "PNG written to gfx/screenshots" << std::endl;
  } else {
    std::cout << "ERROR: Could not write PNG" << std::endl;
  }

  delete[] screen;
};
void createQuadBuffers(unsigned int &VAO, unsigned int &VBO) {
  // Create VAO and VBO for screen filling quad (contour generation)
  float quadVertices[] = {// vertex attributes for a quad that fills the entire
                          // screen in Normalized Device Coordinates.
                          // positions   // texCoords
                          -1.0f, 1.0f, 0.0f, 1.0f,  
                          -1.0f, -1.0f, 0.0f, 0.0f, 
                          1.0f, -1.0f, 1.0f, 0.0f,
                          -1.0f, 1.0f, 0.0f, 1.0f,  
                          1.0f, -1.0f, 1.0f, 0.0f, 
                          1.0f, 1.0f,  1.0f, 1.0f};
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glBindVertexArray(VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        (void *)(2 * sizeof(float)));
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
};

#endif