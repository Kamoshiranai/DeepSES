#ifndef CAMERA_H
#define CAMERA_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>

//
class arcBallCamera {
public:
  int SCR_WIDTH;
  int SCR_HEIGHT;
  // camera Attributes in world space
  glm::vec3 Position;
  glm::vec3 Front;
  glm::vec3 Up;
  glm::vec3 Right;
  glm::vec3 Center;
  glm::vec2 LastPos;
  float Radius;
  // glm::quat view_quat;

  // constructor with vectors
  arcBallCamera(int width, int height, float radius = 1.0f,
                glm::vec3 center = glm::vec3(0.0f, 0.0f, 0.0f),
                glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f)) {
    SCR_WIDTH = width;
    SCR_HEIGHT = height;
    Position = glm::vec3(0.0f, 0.0f, -radius);
    Up = up;
    Center = center;
    Front = -glm::normalize(center - Position);
    Right = glm::normalize(glm::cross(Front, up));
    Radius = radius;
    LastPos = glm::vec2(SCR_WIDTH / 2.0, SCR_HEIGHT / 2.0);

    // view_quat = glm::quat_cast(glm::lookAt(Position, center, up));
  }

  // returns the view matrix calculated using the quaternion -> this is a
  // rotation only
  glm::mat4 GetViewMatrix() { return glm::lookAt(Position, Center, Up); }

  // Set view matrix given a chimerax view matrix
  void SetViewMatrix(float (&arr)[12], glm::vec3 translate) {
    glm::mat4 View = glm::mat4(1.0f);
    // Insert rotation matrix into the top-left 3x3 submatrix
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        View[j][i] = arr[(i * 4) + j];
      }
      Position[i] = View[3][i];
    }
    Position -= translate;

    Right[0] = View[0][0];
    Right[1] = View[0][1];
    Right[2] = View[0][2];

    Up[0] = View[1][0];
    Up[1] = View[1][1];
    Up[2] = View[1][2];

    Front[0] = View[2][0];
    Front[1] = View[2][1];
    Front[2] = View[2][2];
    Center = glm::normalize(Position + Front);
    Position = Position + Radius * Front;
  }
  glm::vec3 getPosition() { return Position; }
  glm::vec3 getFront() { return Front; }
  glm::vec3 getRight() { return Right; }
  glm::vec3 getUp() { return Up; }
  float getRadius() { return Radius; }
  void setPos(int x, int y) { LastPos = glm::vec2(x, y); }
  void setRadius(float radius) {
    Position += Front * (Radius - radius);
    Radius = radius;
  }
  void setScreenSize(int width, int height) {
    SCR_WIDTH = width;
    SCR_HEIGHT = height;
  }
  void rotate_angle_axis(float angle, glm::vec3 axis) {
    glm::quat rot = glm::angleAxis(angle, axis);

    // view_quat = view_quat * rot;
    Front = glm::normalize(rot * Front);
    Right = glm::normalize(rot * Right);
    Up = glm::cross(Right, Front);
    Position = Center - Front * Radius;
    // View = glm::lookAt(Position, Center, Up);
  }

  // Perform arcball rotation. Input is the mouse position
  void rotate(int x, int y) {
    glm::vec3 lastArcPos = get_arcball_vector(LastPos.x, LastPos.y);
    glm::vec3 curArcPos = get_arcball_vector(x, y);
    float angle = acos(std::min(1.0f, glm::dot(lastArcPos, curArcPos)));
    glm::vec3 rotationAxis = glm::cross(lastArcPos, curArcPos);
    glm::mat4 viewMatrix = GetViewMatrix();
    glm::mat3 view2world =
        glm::inverse(viewMatrix); // convert axis to view coordinates
    glm::vec3 axis_in_world_coord = glm::normalize(view2world * rotationAxis);
    glm::quat rot = glm::angleAxis(angle, axis_in_world_coord);

    // view_quat = view_quat * rot;
    Front = glm::normalize(rot * Front);
    Right = glm::normalize(rot * Right);
    Up = glm::cross(Right, Front);
    Position = Center - Front * Radius;
  }

  glm::vec3 get_arcball_vector(
      int x,
      int y) { // see
               // https://en.wikibooks.org/wiki/OpenGL_Programming/Modern_OpenGL_Tutorial_Arcball
    glm::vec3 P = glm::vec3(-1.0 * (2.0 * x / SCR_WIDTH - 1.0),
                            1.0 * (1.0 * y / SCR_HEIGHT * 2 - 1.0), 0);
    float OP_squared =
        P.x * P.x + P.y * P.y; // OP is the vector from the origin to point P
    if (OP_squared <= 1 * 1)
      P.z = sqrt(1 * 1 - OP_squared); // Get z component using Pythagoras
    else
      P = glm::normalize(P); // nearest point
    return P;
  }
};

#endif