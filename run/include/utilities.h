#ifndef UTILITIES_H
#define UTILITIES_H

#include <glad/glad.h>

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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