#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <arcballCamera/arcballCamera.h>
#include <learnopengl/compute_shader.h>
#include <learnopengl/shader.h>
#include <uncertainty/cif+pdbLoader.h>
#include <utilities.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image/stb_image_write.h>

// DEBUG
// // Callback function to handle debug messages
// const char* GetDebugSource(GLenum source) {
//   switch (source) {
//       case GL_DEBUG_SOURCE_API: return "API";
//       case GL_DEBUG_SOURCE_WINDOW_SYSTEM: return "Window System";
//       case GL_DEBUG_SOURCE_SHADER_COMPILER: return "Shader Compiler";
//       case GL_DEBUG_SOURCE_THIRD_PARTY: return "Third Party";
//       case GL_DEBUG_SOURCE_APPLICATION: return "Application";
//       case GL_DEBUG_SOURCE_OTHER: return "Other";
//       default: return "Unknown";
//   }
// }

// const char* GetDebugType(GLenum type) {
//   switch (type) {
//       case GL_DEBUG_TYPE_ERROR: return "Error";
//       case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: return "Deprecated Behavior";
//       case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR: return "Undefined Behavior";
//       case GL_DEBUG_TYPE_PERFORMANCE: return "Performance";
//       case GL_DEBUG_TYPE_OTHER: return "Other";
//       default: return "Unknown";
//   }
// }

// const char* GetDebugSeverity(GLenum severity) {
//   switch (severity) {
//       case GL_DEBUG_SEVERITY_HIGH: return "High";
//       case GL_DEBUG_SEVERITY_MEDIUM: return "Medium";
//       case GL_DEBUG_SEVERITY_LOW: return "Low";
//       case GL_DEBUG_SEVERITY_NOTIFICATION: return "Notification";
//       default: return "Unknown";
//   }
// }

// // The callback function
// void GLAPIENTRY MessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity,
//                               GLsizei length, const GLchar* message, const void* userParam)
// {
//   std::cerr << "OpenGL Debug Message:" << std::endl;
//   std::cerr << "Source: " << GetDebugSource(source) << std::endl;
//   std::cerr << "Type: " << GetDebugType(type) << std::endl;
//   std::cerr << "ID: " << id << std::endl;
//   std::cerr << "Severity: " << GetDebugSeverity(severity) << std::endl;
//   std::cerr << "Message: " << message << std::endl;
// }

//NOTE: helper functions
std::string getFileExtension(const std::string& filename) {
  size_t dotPos = filename.find_last_of('.');
  
  if (dotPos == std::string::npos) return ""; // No extension found

  std::string ext = filename.substr(dotPos);

  // Special case: Handle .gz and similar compressed extensions
  if (ext == ".gz" || ext == ".bz2" || ext == ".zip") {
      size_t secondDotPos = filename.find_last_of('.', dotPos - 1);
      if (secondDotPos == std::string::npos) return ""; // No second extension found
      return filename.substr(secondDotPos+1, dotPos - (secondDotPos+1)); // Extract the real extension before .gz
  }

  return ext.substr(1); // Return normal extension (without ".") if not compressed
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void key_callback(GLFWwindow *window, int key, int scancode, int action,
                  int mods);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
glm::vec3 get_arcball_vector(int x, int y);

// settings

// Orthographic cameras field width chimerax (use camera command to show)
float w = 100.59;

// Scr_WIDTH and HEIGHT need to be doubled from the chimerax values (command
// (window size))
unsigned int SCR_WIDTH = 1920;
unsigned int SCR_HEIGHT = 1080;

// view matrix from chimerax (command 'view matrix' under view matrix camera)
bool setView = false;
float view_arr[12] = {0.2879,   0.92166, 0.26011,  23.992,  0.66501,  0.0030433,
                      -0.74683, -75.643, -0.68911, 0.38799, -0.61204, -55.885};

arcBallCamera camera(SCR_WIDTH, SCR_HEIGHT);
enum Encoding { Noise, Hull, Contours, Texture, Contour_Only, No };
Encoding uncertainty = No;
float zoom = 45.0f;
float r_probe = 1.4;
bool all_atoms = true; // Should all atoms be considered for the surface?
int num_atoms = 100;   // If not all atoms: maximum number of atoms considered
bool bool_uniform_color =
    true; // Color uniformly (true)? Or per atom type (false)?
glm::vec3 uniform_color = glm::vec3(26, 41, 88) / 100.0f;
// std::string protein_file = "../../../data/pdb/1adn_single.pdb";

int res = 512;
const glm::ivec3 dim_grid = glm::ivec3(res, res, res);
bool camera_changed = true;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: ./interactive-vdw <file_path>" << std::endl;
    return 1;
  }

  std::string protein_file = argv[1];
  std::string filetype = getFileExtension(protein_file);

  // ------------------------------------------------------------------------------
  //  OpenGl boilerplate
  // ------------------------------------------------------------------------------
  glfwInit();
  const char *glsl_version = "#version 430";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow *window =
      glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "SES SDF", NULL, NULL);
  if (window == NULL) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetCursorPosCallback(window, mouse_callback);
  glfwSetScrollCallback(window, scroll_callback);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
  glfwSetKeyCallback(window, key_callback);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  //DEBUG
  // // Enable OpenGL debug output
  // glEnable(GL_DEBUG_OUTPUT);
  // glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);  // Optional: ensures messages are delivered synchronously

  // // Set the debug message callback function
  // glDebugMessageCallback(MessageCallback, 0);

  // Debug get max size of 3D texture
  // GLint maxTexSize;
  // glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &maxTexSize);
  // std::cout << "Max 3D Texture Size: " << maxTexSize << std::endl;


  // ------------------------------------------------------------------------------
  // Load molecule
  // ------------------------------------------------------------------------------
  Protein Coords;
  if (!Coords.Load(protein_file, filetype)) {
    std::cerr << "Error during loading of file " << protein_file << std::endl;
    return 0;
  }

  std::cout << "read file sucessfully, num atoms: " << Coords.numAtoms << std::endl;

  if (all_atoms) {
    num_atoms = Coords.numAtoms;
  }

  // Find bounding box
  auto [xmin, xmax, ymin, ymax, zmin, zmax] = Coords.ComputeBounds(num_atoms);

  float xlength = xmax - xmin;
  float ylength = ymax - ymin;
  float zlength = zmax - zmin;

  // Compute extents of the molecule for creating the grids
  float scale_model = std::max(std::max(xlength, ylength), zlength);
  scale_model += Coords.LargestVdwRadius() * 2 + r_probe * 2;
  glm::vec3 dim_grid_prelim = glm::vec3(scale_model, scale_model, scale_model);

  // Get resolution for specified grid size
  float resolution = dim_grid_prelim.x / glm::vec3(dim_grid).x;

  // Grid for storing neighbors info
  float n_resolution = r_probe + Coords.LargestVdwRadius();
  glm::ivec3 n_dim_grid =
      glm::ivec3(glm::ceil(dim_grid_prelim * (1.0f / n_resolution)));

  // ------------------------------------------------------------------------------
  // Place molecule and camera in global space
  // ------------------------------------------------------------------------------

  //  This is the distance to the center of the molecule in which the camera
  //  be placed
  camera.setRadius(scale_model * 1.5);

  // Put the model at the scene origin
  glm::mat4 model = glm::mat4(1.0f);
  glm::vec3 translate =
      glm::vec3(-xmin - xlength / 2, -ymin - ylength / 2, -zmin - zlength / 2);
  model = glm::translate(model, translate);
  bool model_changed = true;

  for (int i = 0; i < num_atoms; ++i) {
    Coords.atoms[i].coords =
        glm::vec3(model * glm::vec4(glm::vec3(Coords.atoms[i].coords), 1.0));
  }
  glm::mat4 view = glm::mat4(1.0f);
  glm::vec3 camera_pos;
  glm::vec3 camera_front;
  if (setView) {

    // Get chimerax view matrix array into proper matrix format
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        view[i][j] = view_arr[(i * 4) + j];
      }
      // camera_pos[i] = view_arr[(i * 4) + 3];
      view[3][i] = -view_arr[(i * 4) + 3];
    }

    // Extract basis vectors
    camera_pos.x = -view[3][0];
    camera_pos.y = -view[3][1];
    camera_pos.z = -view[3][2];

    camera_front.x = -view[0][2];
    camera_front.y = -view[1][2];
    camera_front.z = -view[2][2];

    glm::vec3 up;
    up.x = view[0][1];
    up.y = view[1][1];
    up.z = view[2][1];

    // Transform into centered molecule space
    camera_pos = glm::vec3(model * glm::vec4(camera_pos, 1.0));
    camera_front = glm::normalize(camera_front);

    // Calculate view matrix for centered molecule space
    view = glm::lookAt(camera_pos, camera_pos + camera_front, up);

    camera.setRadius(length(camera_pos));
  }

  // ------------------------------------------------------------------------------
  // Create SSBOs
  // ------------------------------------------------------------------------------

  // Holds the atom information
  unsigned int atoms_ssbo;
  glGenBuffers(1, &atoms_ssbo);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, atoms_ssbo);
  glBufferData(GL_SHADER_STORAGE_BUFFER, num_atoms * 8 * sizeof(float),
               Coords.atoms.data(), GL_STATIC_DRAW);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, atoms_ssbo);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

  // Index buffer that is used to check which items lie on the boundary
  unsigned int index_ssbo;
  const int index_array_size = dim_grid.x / 8 * dim_grid.y / 8 * dim_grid.z / 8;
  std::vector<GLuint> index_data(index_array_size);
  for (int i = 0; i < index_array_size; ++i) {
    // Initialize with large number to be able to later exclude non-boundary
    // points from phase2
    index_data.at(i) = index_array_size * 2;
  }

  glGenBuffers(1, &index_ssbo);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, index_ssbo);
  glBufferData(GL_SHADER_STORAGE_BUFFER, index_array_size * sizeof(int),
               index_data.data(), GL_STATIC_DRAW);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, index_ssbo);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 1);

  // SSBOs for the neighborhood Grid see paper by Green
  // https://developer.download.nvidia.cn/assets/cuda/files/particles.pdf
  int cell_size = 10; //NOTE: 6 was too small
  int n_grid_size = n_dim_grid.x * n_dim_grid.y * n_dim_grid.z;
  unsigned int grid_counters;
  // Initialize counters and cells to 0
  // use dynamic allocation, as n_grid_size may not be known at compile time
  std::vector<int> counters(n_grid_size, 0);
  std::vector<int> cells(n_grid_size * cell_size, 0);
  glGenBuffers(1, &grid_counters);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid_counters);
  glBufferData(GL_SHADER_STORAGE_BUFFER, n_grid_size * sizeof(int),
               &counters[0], GL_STREAM_COPY);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, grid_counters);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 2);

  unsigned int grid_cells;
  glGenBuffers(1, &grid_cells);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid_cells);
  glBufferData(GL_SHADER_STORAGE_BUFFER,
               n_grid_size * cell_size * sizeof(GLuint), &cells[0],
               GL_STREAM_COPY);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, grid_cells);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 3);

  // ------------------------------------------------------------------------------
  // Bind SSBOs
  // ------------------------------------------------------------------------------
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, atoms_ssbo);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, atoms_ssbo);

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, index_ssbo);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, index_ssbo);

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid_counters);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, grid_counters);

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid_cells);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, grid_cells);

  // ------------------------------------------------------------------------------
  // Generate Textures
  // ------------------------------------------------------------------------------
  // Textures to hold the information for the final shading
  GLuint tex_pos, tex_normal, tex_color;
  glGenTextures(1, &tex_pos);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, tex_pos);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA,
               GL_FLOAT, NULL);
  glBindImageTexture(0, tex_pos, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

  glGenTextures(1, &tex_normal);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, tex_normal);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA,
               GL_FLOAT, NULL);
  glBindImageTexture(1, tex_normal, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

  glGenTextures(1, &tex_color);
  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, tex_color);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA,
               GL_FLOAT, NULL);
  glBindImageTexture(2, tex_color, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

  // 3D Texture for the distance field stores color of the closest atom and
  // distance to surface as vec4: vec4(vec3(color), float(distance))
  GLuint tex_dist_field;
  glGenTextures(1, &tex_dist_field);
  glActiveTexture(GL_TEXTURE0); //NOTE activate texture unit
  glBindTexture(GL_TEXTURE_3D, tex_dist_field); //NOTE: bind texture to this texture unit
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, dim_grid.x, dim_grid.y, dim_grid.z,
               0, GL_RGBA, GL_FLOAT, NULL);
  glBindImageTexture(0, tex_dist_field, 0, GL_TRUE, 0, GL_READ_WRITE,
                     GL_RGBA32F); //NOTE: bind texture to this (texture) image unit

  // ------------------------------------------------------------------------------
  // Bind Textures
  // ------------------------------------------------------------------------------

  glActiveTexture(GL_TEXTURE1);
  glBindImageTexture(1, tex_pos, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);
  glActiveTexture(GL_TEXTURE2);
  glBindImageTexture(2, tex_normal, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);
  glActiveTexture(GL_TEXTURE3);
  glBindImageTexture(3, tex_color, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);

  // ------------------------------------------------------------------------------
  // shaders
  // ------------------------------------------------------------------------------

  // Calculate the neighborhood grid
  cShader neighbors_shader("shaders/neighbors.cs");
  // Calculate vdw
  cShader phase1_shader("shaders/phase1_vdw_interactive.cs");

  // Calculate color, position, and normals
  cShader raymarch_shader("shaders/raymarch-molecule.cs");
    // Shading
  Shader phong_shader("shaders/passthrough.vs", "shaders/phong.fs");

  neighbors_shader.use();
  // Size of the neighbors grid
  neighbors_shader.setVec3("n_dims", n_dim_grid);
  // Resolution of the neighbors grid
  neighbors_shader.setFloat("n_resolution", n_resolution);
  // number of atoms that can be stored in a single cell
  neighbors_shader.setInt("cell_size", cell_size);

  phase1_shader.use();
  phase1_shader.setFloat("r_probe", r_probe);
  // Resolution of the sdf grid
  phase1_shader.setFloat("resolution", resolution);
  phase1_shader.setFloat("n_resolution", n_resolution);
  phase1_shader.setVec3("n_dims", n_dim_grid);
  phase1_shader.setInt("cell_size", cell_size);

  unsigned int quadVAO, quadVBO;
  createQuadBuffers(quadVAO, quadVBO);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(quadVAO);
  glDisable(GL_DEPTH_TEST);

  // ------------------------------------------------------------------------------
  // Renderloop
  // ------------------------------------------------------------------------------
  double lastTime = glfwGetTime();
  int nbFrames = 0;
  raymarch_shader->use();
  raymarch_shader->setInt("num_atoms", num_atoms);
  raymarch_shader->setFloat("r_probe", r_probe);

  raymarch_shader->setVec3("dims", dim_grid);
  raymarch_shader->setFloat("grid_res", resolution);

  phong_shader->use();
  phong_shader->setBool("bool_uniform_color", bool_uniform_color);
  phong_shader->setInt("tex_pos", 0);
  phong_shader->setInt("tex_normal", 1);
  phong_shader->setInt("tex_color", 2);
  phong_shader->setVec3("uniform_color", uniform_color);

  // field width from chimerax needs to be halfed
  w *= 0.5;

  while (!glfwWindowShouldClose(window)) {

    // ------------------------------------------------------------------------------
    // Timings
    // ------------------------------------------------------------------------------

    // Measure speed
    double currentTime = glfwGetTime();
    nbFrames++;
    double elapsed_time = currentTime - lastTime;
    if (elapsed_time >= 0.5) { // If last prinf() was more than 1 sec ago
      // printf and reset timer
      std::cout << elapsed_time * 1000.0 / double(nbFrames) << " ms/frame,"
                << double(nbFrames) / elapsed_time << " fps" << std::endl;
      nbFrames = 0;
      lastTime = currentTime;
      std::cout << "Screen size" << SCR_WIDTH << "x" << SCR_HEIGHT << std::endl;
    }
    // ------------------------------------------------------------------------------
    // Camera
    // ------------------------------------------------------------------------------

    // Uncomment for rotating molecule
    // camera.rotate_angle_axis(glm::radians(1.0), glm::vec3(0, 1, 0));
    camera_changed = true;

    if (!setView) {
      view = camera.GetViewMatrix();
      camera_pos = camera.getPosition();
      w = camera.getRadius() * 0.5f;
      camera_front = glm::normalize(camera.getFront());
    }
    float aspect = (float)SCR_WIDTH / SCR_HEIGHT;
    glm::mat4 projection =
        glm::ortho(-w, w, -w / aspect, w / aspect, -200.0f, 200.0f); //NOTE: for orthographic projection

    if (camera_changed) {
      if (model_changed) {
        // ------------------------------------------------------------------------------
        // Compute neighbors grid
        // ------------------------------------------------------------------------------
        neighbors_shader.use();

        glDispatchCompute(num_atoms, 1, 1);

        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid_counters);

        // ------------------------------------------------------------------------------
        // Phase 1: Probe intersection
        // ------------------------------------------------------------------------------
        phase1_shader.use();

        glDispatchCompute(dim_grid.x / 8, dim_grid.y / 8, dim_grid.z / 8);

        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

        // ------------------------------------------------------------------------------
        // Reset neighbors buffers to 0 for the next time step
        // ------------------------------------------------------------------------------
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid_counters);
        int zero_int = 0;
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32I, GL_RED_INTEGER, GL_INT, &zero_int);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid_cells);
        GLuint zero_uint = 0;
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero_uint);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

        model_changed = true;
      }

      // ------------------------------------------------------------------------------
      // Raymarch the SDF
      // ------------------------------------------------------------------------------
      raymarch_shader->use();
      raymarch_shader->setMat4("view", view);
      raymarch_shader->setMat4("projection", projection);
      raymarch_shader->setVec3("camera_pos", camera_pos);
      raymarch_shader->setVec3("camera_front", camera_front);
      raymarch_shader->setVec2("resolution", glm::vec2(SCR_WIDTH, SCR_HEIGHT));
      raymarch_shader->setInt("grid", 0);
      raymarch_shader->setInt("grid_b_factor", 4);

      glDispatchCompute((GLuint)SCR_WIDTH / 4, (GLuint)SCR_HEIGHT / 4, 1);
      glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

      // ------------------------------------------------------------------------------
      // Lighting Pass, render to screen
      // ------------------------------------------------------------------------------
      glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      phong_shader->use();
      phong_shader->setVec3("camera_pos", camera_pos);
      glDrawArrays(GL_TRIANGLES, 0, 6);
      glDisable(GL_BLEND);

      glfwSwapBuffers(window);
      camera_changed = false;
    }
    glfwPollEvents();
  }

  // Clean up
  glDeleteVertexArrays(1, &quadVAO);
  glDeleteBuffers(1, &quadVBO);

  delete raymarch_shader;
  delete phong_shader;

  glfwTerminate();
  return 0;
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
  SCR_HEIGHT = height;
  SCR_WIDTH = width;
  camera.setScreenSize(width, height);
  glActiveTexture(GL_TEXTURE0);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA,
               GL_FLOAT, NULL);
  glActiveTexture(GL_TEXTURE1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA,
               GL_FLOAT, NULL);
  glActiveTexture(GL_TEXTURE2);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA,
               GL_FLOAT, NULL);
  glActiveTexture(GL_TEXTURE0);
  camera_changed;
}

// rotate the object with the mouse using and arcball transformation
void mouse_callback(GLFWwindow *window, double xpos, double ypos) {
  // rotate the object with the mouse using and arcball transformation
  int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
  if (state == GLFW_PRESS) {
    camera.rotate(xpos, ypos);
  }
  camera.setPos(xpos, ypos);
  camera_changed = true;
}
void key_callback(GLFWwindow *window, int key, int scancode, int action,
                  int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);
}

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
  // zoom
  camera.setRadius(camera.getRadius() + 4.0 * yoffset);
  camera_changed = true;
}