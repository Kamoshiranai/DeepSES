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

#include <chrono> // for fps calculation
#include <numeric> // for std::accumulate
#include <cmath> // Required for std::sqrt and std::pow
#include <iomanip> // For std::fixed, std::setprecision

// cuda kernels for reshaping buffer and filtering out relevant patches
#include <kernels.h>
#include "thrust_wrapper.cuh"

#include <NvInfer.h>          // Core TensorRT API
#include <NvInferRuntime.h>   // For runtime functionalities
#include <NvOnnxParser.h>     // If parsing ONNX models
//#include <NvInferPlugin.h>    // If using TensorRT plugins
#include <cuda_runtime.h>     // For CUDA memory management
#include <cuda_gl_interop.h>  // For using openGL objects as cuda ressources

#include <nvml.h> // for gpu memory

#include <iostream>           // For logging and debugging
#include <vector>             // For handling inputs/outputs
#include <vector_types.h> // For int3, float3, etc.
#include <vector_functions.hpp> // For make_int3, floor, etc.
#include <stdio.h>          // For printf

#include <fstream>   // for std::ofstream
#include <string>    // for std::string

//DEBUG: printing type of buffer
#include <typeinfo>

#ifdef _WIN32
    // Windows - No standard demangling without extra libs/code
    // Placeholder or use typeid directly
    std::string demangle(const char* mangled_name) {
        // Basic fallback: return the mangled name
        return (mangled_name ? mangled_name : "<null>");
        // OR: Implement using DbgHelp.h/UnDecorateSymbolName if needed (complex)
}
#else
    // Linux/macOS using GCC/Clang
    #include <cxxabi.h>
    #include <memory> // For unique_ptr
    #include <cstdlib> // For free


    // Helper function to demangle type names (makes output more readable)
    std::string demangle(const char* name) {
        int status = -1; // Some arbitrary value to eliminate compiler warning

        // Use smart pointer for automatic memory management
        std::unique_ptr<char, void(*)(void*)> res{
            abi::__cxa_demangle(name, NULL, NULL, &status),
            std::free
        };

        return (status == 0) ? res.get() : name;
    }
#endif

// CUDA error checking macro
#define checkCudaErrors(call)                                                  \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

void checkCudaErrorCode(cudaError_t code) {
  if (code != cudaSuccess) {
    std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + " (" + cudaGetErrorName(code) +
                          "), with message: " + cudaGetErrorString(code);
    throw std::runtime_error(errMsg);
  }
}

// void printCudaDeviceLimits() {
//   int deviceID;
//   cudaError_t err = cudaGetDevice(&deviceID); // Get the ID of the current device
//   if (err != cudaSuccess) {
//       fprintf(stderr, "Error getting current CUDA device: %s\n", cudaGetErrorString(err));
//       return;
//   }

//   cudaDeviceProp deviceProperties;
//   err = cudaGetDeviceProperties(&deviceProperties, deviceID); // Get properties for this device
//   if (err != cudaSuccess) {
//       fprintf(stderr, "Error getting properties for device %d: %s\n", deviceID, cudaGetErrorString(err));
//       return;
//   }

//   printf("--- CUDA Device %d (%s) Limits ---\n",
//          deviceID, deviceProperties.name);

//   // Maximum threads per block (total)
//   printf("Max Threads Per Block:        %d\n", deviceProperties.maxThreadsPerBlock);

//   // Maximum dimensions of a thread block (X, Y, Z)
//   printf("Max Block Dimensions (X,Y,Z): (%d, %d, %d)\n",
//          deviceProperties.maxThreadsDim[0],
//          deviceProperties.maxThreadsDim[1],
//          deviceProperties.maxThreadsDim[2]);

//   // Maximum dimensions of a grid of blocks (X, Y, Z)
//   printf("Max Grid Dimensions (X,Y,Z):  (%d, %d, %d)\n",
//          deviceProperties.maxpatches_per_dim[0],
//          deviceProperties.maxpatches_per_dim[1],
//          deviceProperties.maxpatches_per_dim[2]);

//   printf("---------------------------------------\n");
// }

//DEBUG
// // Function to visualize a small portion of the 3D buffer
// template <typename T>
// void printBuffer(std::vector<T> hostBuffer, int width, int height, int depth) {
//   // --- Print Type Information ---
//   // Get the mangled name first
//   const char* mangled_name = typeid(T).name();
//   // Demangle it for readability
//   std::string readable_name = demangle(mangled_name);
//   std::cout << "--- Buffer Data (Type: " << readable_name << ") ---" << std::endl;
//   // Print buffer data
//   for (int d = 0; d < depth; d++) {  // Coul only print a few slices
//       //std::cout << "Slice " << d << ":\n";
//       for (int h = 0; h < height; h++) {
//           for (int w = 0; w < width; w++) {
//             std::cout << hostBuffer[d * width * height + h * width + w] << " ";
//           }
//           std::cout << "\n";
//       }
//       std::cout << "\n";
//   }
// }

// template <typename T>
// void visualizeCudaBuffer(const T* d_buffer, int width, int height, int depth, cudaStream_t stream) {
//   size_t bufferSize = width * height * depth * sizeof(T);
//   std::vector<T> h_buffer(width * height * depth);

//   // Copy device buffer to host
//   cudaMemcpyAsync(h_buffer.data(), d_buffer, bufferSize, cudaMemcpyDeviceToHost, stream);
//   checkCudaErrors(cudaStreamSynchronize(stream));

//   // Print part of the buffer
//   printBuffer(h_buffer, width, height, depth);

// }

// // Function to visualize a small portion of the 3D buffer
// template <typename T>
// void printIndices(std::vector<T> hostBuffer, std::vector<int> hostIndexBuffer, int width, int height, int depth) {
//   // --- Print Type Information ---
//   // Get the mangled name first
//   const char* mangled_name = typeid(T).name();
//   // Demangle it for readability
//   std::string readable_name = demangle(mangled_name);
//   std::cout << "--- Buffer Data (Type: " << readable_name << ") ---" << std::endl;
//   // Print buffer data
//   for (int d = 0; d < depth; d++) {  // Coul only print a few slices
//       //std::cout << "Slice " << d << ":\n";
//       for (int h = 0; h < height; h++) {
//           for (int w = 0; w < width; w++) {
//             if (hostIndexBuffer[d * width * height + h * width + w] == 1) {
//               std::cout << hostBuffer[d * width * height + h * width + w] << " ";
//             }
//           }
//           std::cout << "\n";
//       }
//       std::cout << "\n";
//   }
// }

// template <typename T>
// void visualizeCudaIndices(const T* d_buffer, const int* indexBuffer, int width, int height, int depth, cudaStream_t stream) {
//   size_t bufferSize = width * height * depth * sizeof(T);
//   std::vector<T> h_buffer(width * height * depth);
//   std::vector<T> h_indexBuffer(width * height * depth);

//   // Copy device buffer to host
//   cudaMemcpyAsync(h_buffer.data(), d_buffer, bufferSize, cudaMemcpyDeviceToHost, stream);
//   cudaMemcpyAsync(h_indexBuffer.data(), indexBuffer, bufferSize, cudaMemcpyDeviceToHost, stream);
//   checkCudaErrors(cudaStreamSynchronize(stream));

//   // Print part of the buffer
//   printIndices(h_buffer, h_indexBuffer, width, height, depth);

// }

// void copyAndPrintBuffer(cudaPitchedPtr d_batchedInputBuffer, int batch, int voxels_per_patch_dim) {
//   size_t pitch = d_batchedInputBuffer.pitch;
//   size_t rowSize = voxels_per_patch_dim * sizeof(float);
//   size_t height = voxels_per_patch_dim;
//   size_t depth = batch * voxels_per_patch_dim;

//   float* h_buffer = new float[batch * voxels_per_patch_dim * voxels_per_patch_dim * voxels_per_patch_dim];

//   // Copy row by row
//   for (int z = 0; z < depth; ++z) {
//       for (int y = 0; y < height; ++y) {
//           cudaMemcpy(h_buffer + z * voxels_per_patch_dim * voxels_per_patch_dim + y * voxels_per_patch_dim, 
//                      (char*)d_batchedInputBuffer.ptr + z * pitch * height + y * pitch, 
//                      rowSize, cudaMemcpyDeviceToHost);
//       }
//   }

//   // Print some values
//   for (int i = 0; i < voxels_per_patch_dim; i++) {
//       std::cout << h_buffer[i] << " ";
//   }
//   std::cout << std::endl;

//   delete[] h_buffer;
// }

// void printRaymarchParams(const RaymarchParams& p) {
//   printf("\n--- RaymarchParams Snapshot ---\n");

//   // --- Input Data & Grid Info ---
//   printf("  gridTexture Handle:    %llu (Opaque CUDA handle)\n", p.gridTexture); // cudaTextureObject_t is unsigned long long
//   printf("  gridDimsVoxels:              (%d, %d, %d)\n", p.gridDimsVoxels.x, p.gridDimsVoxels.y, p.gridDimsVoxels.z);
//   printf("  gridRes:               %.6f\n", p.gridRes);

//   // --- Patch Info ---
//   printf("  patchDimsVoxels:       (%d, %d, %d)\n", p.patchDimsVoxels.x, p.patchDimsVoxels.y, p.patchDimsVoxels.z);
//   printf("  patches_per_dim:         (%d, %d, %d)\n", p.patchesPerDim.x, p.patchesPerDim.y, p.patchesPerDim.z);
//   printf("  totalPatches:          %d\n", p.totalPatches);

//   // --- Camera Info ---
//   printf("  cameraPos:             (%.4f, %.4f, %.4f)\n", p.cameraPos.x, p.cameraPos.y, p.cameraPos.z);
//   printf("  cameraFront:           (%.4f, %.4f, %.4f)\n", p.cameraFront.x, p.cameraFront.y, p.cameraFront.z);
//   printf("  invViewProj Matrix:\n");
//   // Assuming float4x4 struct contains float4 rows[4]
//   // Adjust if using float v[16] or float v[4][4]
//   for (int i = 0; i < 4; ++i) {
//       printf("    row %d: [%+8.4f %+8.4f %+8.4f %+8.4f]\n", i,
//              p.invViewProj.rows[i].x,
//              p.invViewProj.rows[i].y,
//              p.invViewProj.rows[i].z,
//              p.invViewProj.rows[i].w);
//   }

//   // --- Screen Info ---
//   printf("  resolution:            (%u, %u)\n", p.resolution.x, p.resolution.y);

//   // --- Raymarching Params ---
//   printf("  epsilon:               %g\n", p.epsilon); // %g is often good for floats
//   printf("  maxSteps:              %d\n", p.maxSteps);

//   // --- Output Buffer ---
//   printf("  visitedPatches Ptr:    %p (Device pointer)\n", (void*)p.visitedPatches); // Cast to void* for %p

//   printf("-----------------------------\n\n");
// }

// DEBUG:
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

// // Helper function to check for OpenGL errors
// // Call this after potentially error-prone sections
// inline void checkGLError(const char* checkpoint_name = "") {
//     GLenum error;
//     bool errors_found = false;
//     while ((error = glGetError()) != GL_NO_ERROR) {
//         errors_found = true;
//         std::cerr << "OpenGL Error";
//         if (checkpoint_name && checkpoint_name[0] != '\0') {
//             std::cerr << " at " << checkpoint_name;
//         }
//         std::cerr << ": ";
//         switch (error) {
//             case GL_INVALID_ENUM:      std::cerr << "GL_INVALID_ENUM"; break;
//             case GL_INVALID_VALUE:     std::cerr << "GL_INVALID_VALUE"; break;
//             case GL_INVALID_OPERATION: std::cerr << "GL_INVALID_OPERATION"; break;
//             case GL_STACK_OVERFLOW:    std::cerr << "GL_STACK_OVERFLOW"; break;
//             case GL_STACK_UNDERFLOW:   std::cerr << "GL_STACK_UNDERFLOW"; break;
//             case GL_OUT_OF_MEMORY:     std::cerr << "GL_OUT_OF_MEMORY"; break;
//             case GL_INVALID_FRAMEBUFFER_OPERATION: std::cerr << "GL_INVALID_FRAMEBUFFER_OPERATION"; break;
//             default:                   std::cerr << "Unknown error code: " << error; break;
//         }
//         std::cerr << std::endl;
//     }
//     // Optional: Assert or throw if an error was found to halt execution immediately
//     // if (errors_found) {
//     //     throw std::runtime_error("OpenGL Error detected!");
//     // }
//   }

// NOTE: helper functions
std::string getFileExtension(const std::string& filename) {
  size_t dotPos = filename.find_last_of('.');
  
  if (dotPos == std::string::npos) return ""; // No extension found

  std::string ext = filename.substr(dotPos+1);

  // Special case: Handle .gz and similar compressed extensions
  if (ext == "gz" || ext == "bz2" || ext == "zip") {
      size_t secondDotPos = filename.find_last_of('.', dotPos - 1);
      if (secondDotPos == std::string::npos) return ""; // No second extension found
      return filename.substr(secondDotPos+1, dotPos - (secondDotPos+1)); // Extract the real extension before .gz
  }

  return ext.substr(1); // Return normal extension (without ".") if not compressed
}

std::string get_filename_from_path(const std::string& path) {
  // Find the position of the last directory separator
  size_t pos = path.find_last_of("/\\");  // Handles both '/' and '\\' for cross-platform compatibility
  std::string filename = (pos == std::string::npos) ? path : path.substr(pos + 1);  // Extract filename

  // Handle known compressed extensions (e.g., .cif.gz, .tar.gz)
  std::vector<std::string> compressed_extensions = {".gz", ".bz2", ".xz", ".zip", ".tar.gz", ".tar.bz2", ".tar.xz"};
  for (const auto& ext : compressed_extensions) {
      if (filename.size() > ext.size() && filename.substr(filename.size() - ext.size()) == ext) {
          filename = filename.substr(0, filename.size() - ext.size());  // Remove the compressed extension
          break;  // Stop after removing the first match
      }
  }

  // Find the position of the last dot (.) to remove the main extension
  size_t dot_pos = filename.find_last_of(".");
  if (dot_pos == std::string::npos) {
      return filename;  // No extension, return filename as is
  }
  
  return filename.substr(0, dot_pos);  // Extract filename without the extension
}

// memory usage:
void printNvmlMem(const char* tag) {
  nvmlInit();
  nvmlDevice_t dev;
  nvmlDeviceGetHandleByIndex(0, &dev);  // GPU 0
  nvmlMemory_t memInfo;
  nvmlDeviceGetMemoryInfo(dev, &memInfo);
  std::cout << tag
            << " — Used: " << (memInfo.used / (1024.0*1024.0)) << " MB, "
            << "Free: " << (memInfo.free / (1024.0*1024.0)) << " MB, "
            << "Total: " << (memInfo.total / (1024.0*1024.0)) << " MB\n";
  nvmlShutdown();
}

// NOTE: profiler for TensorRT execution context
class MyProfiler : public nvinfer1::IProfiler
{
public:
    void reportLayerTime(const char* layerName, float ms) noexcept override
    {
        mProfile[layerName] = ms;
        mTotalTime += ms;
    }

    void printLayerTimes() const
    {
        // for (const auto& entry : mProfile)
        // {
        //     std::cout << "Layer " << entry.first << " - " << entry.second << " ms" << std::endl;
        // }

        std::cout << "=== Total inference time: " << mTotalTime << " ms ===" << std::endl;
    }

    void reset()
    {
        mProfile.clear();
        mTotalTime = 0.0f;
    }

private:
    std::map<std::string, float> mProfile;
    float mTotalTime = 0.0f;
};

// -------------------------------------------------------------
// Set up performance benchmarking
// -------------------------------------------------------------

// --- Configuration ---
const int NUM_FRAMES_DELAY = 36; // Latency: Query results from N-2 frames ago

// --- Global or Class Members ---
std::vector<cudaEvent_t> cudaStartEvents_raymarch(NUM_FRAMES_DELAY);
std::vector<cudaEvent_t> cudaStopEvents_raymarch(NUM_FRAMES_DELAY);

std::vector<cudaEvent_t> cudaStartEvents_gather(NUM_FRAMES_DELAY);
std::vector<cudaEvent_t> cudaStopEvents_gather(NUM_FRAMES_DELAY);

std::vector<cudaEvent_t> cudaStartEvents_inference(NUM_FRAMES_DELAY);
std::vector<cudaEvent_t> cudaStopEvents_inference(NUM_FRAMES_DELAY);

std::vector<cudaEvent_t> cudaStartEvents_scatter(NUM_FRAMES_DELAY);
std::vector<cudaEvent_t> cudaStopEvents_scatter(NUM_FRAMES_DELAY);

std::vector<cudaEvent_t> cudaStartEvents_total(NUM_FRAMES_DELAY);
std::vector<cudaEvent_t> cudaStopEvents_total(NUM_FRAMES_DELAY);

std::vector<GLuint> glQueries(NUM_FRAMES_DELAY);
std::vector<float> cudaFrameTimesMs_raymarch(NUM_FRAMES_DELAY, 0.0f);      // Store results per delay slot
std::vector<float> cudaFrameTimesMs_gather(NUM_FRAMES_DELAY, 0.0f);      // Store results per delay slot
std::vector<float> cudaFrameTimesMs_inference(NUM_FRAMES_DELAY, 0.0f);      // Store results per delay slot
std::vector<float> cudaFrameTimesMs_scatter(NUM_FRAMES_DELAY, 0.0f);      // Store results per delay slot
std::vector<float> cudaFrameTimesMs_total(NUM_FRAMES_DELAY, 0.0f);      // Store results per delay slot
std::vector<GLuint64> glFrameTimesNs(NUM_FRAMES_DELAY, 0); // Store results per delay slot
int currentFrameIndex = 0;

double lastAvgCudaTimeMs_raymarch = 0.0;
double lastAvgCudaTimeMs_gather = 0.0;
double lastAvgCudaTimeMs_inference = 0.0;
double lastAvgCudaTimeMs_scatter = 0.0;
double lastAvgCudaTimeMs_total = 0.0;
double lastAvgGLTimeMs = 0.0;

double stdDevCudaTimeMs_raymarch = 0.0;
double stdDevCudaTimeMs_gather = 0.0;
double stdDevCudaTimeMs_inference = 0.0;
double stdDevCudaTimeMs_scatter = 0.0;
double stdDevCudaTimeMs_total = 0.0;
double stdDevGLTimeMs = 0.0;

// --- Initialization (Before the main loop) ---
void initializeTimers() {
    for (int i = 0; i < NUM_FRAMES_DELAY; ++i) {
        cudaEventCreate(&cudaStartEvents_raymarch[i]);
        cudaEventCreate(&cudaStopEvents_raymarch[i]);
        cudaEventCreate(&cudaStartEvents_gather[i]);
        cudaEventCreate(&cudaStopEvents_gather[i]);
        cudaEventCreate(&cudaStartEvents_inference[i]);
        cudaEventCreate(&cudaStopEvents_inference[i]);
        cudaEventCreate(&cudaStartEvents_scatter[i]);
        cudaEventCreate(&cudaStopEvents_scatter[i]);
        cudaEventCreate(&cudaStartEvents_total[i]);
        cudaEventCreate(&cudaStopEvents_total[i]);
        glGenQueries(1, &glQueries[i]);
        // Initial dummy query to prevent issues on first frame's get
        glBeginQuery(GL_TIME_ELAPSED, glQueries[i]);
        glEndQuery(GL_TIME_ELAPSED);
    }
    // Ensure CUDA context is set up if not done elsewhere
    cudaFree(0);
}

// --- Cleanup (After the main loop) ---
void cleanupTimers() {
    // Important: Ensure GPU is idle before destroying events/queries used recently
    cudaDeviceSynchronize();
    glFinish();

    for (int i = 0; i < NUM_FRAMES_DELAY; ++i) {
        cudaEventDestroy(cudaStartEvents_raymarch[i]);
        cudaEventDestroy(cudaStopEvents_raymarch[i]);
        cudaEventDestroy(cudaStartEvents_gather[i]);
        cudaEventDestroy(cudaStopEvents_gather[i]);
        cudaEventDestroy(cudaStartEvents_inference[i]);
        cudaEventDestroy(cudaStopEvents_inference[i]);
        cudaEventDestroy(cudaStartEvents_scatter[i]);
        cudaEventDestroy(cudaStopEvents_scatter[i]);
        cudaEventDestroy(cudaStartEvents_total[i]);
        cudaEventDestroy(cudaStopEvents_total[i]);
    }
    glDeleteQueries(NUM_FRAMES_DELAY, glQueries.data());
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
float zoom = 45.0f;
float r_probe = 1.4;
bool all_atoms = true; // Should all atoms be considered for the surface?
int num_atoms = 100;   // If not all atoms: maximum number of atoms considered
bool bool_uniform_color = true; // Color uniformly (true)? Or per atom type (false)?
glm::vec3 uniform_color = glm::vec3(26, 41, 88) / 100.0f;

bool camera_changed = true;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: ./interactive-deepses <file_path> <patches_per_dim>" << std::endl;
    return 1;
  }

  std::string protein_file = argv[1];
  std::string filename = get_filename_from_path(protein_file);
  std::string filetype = getFileExtension(protein_file);

  // print gpu memory before render loo
  // printNvmlMem("start");
  // determine gpu memory before render loop
  nvmlInit();
  nvmlDevice_t dev;
  nvmlDeviceGetHandleByIndex(0, &dev);  // GPU 0
  nvmlMemory_t memInfo;
  nvmlDeviceGetMemoryInfo(dev, &memInfo);
  float gpu_memory_start = memInfo.used / (1024.0*1024.0);
  nvmlShutdown();

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

  std::cout << Coords.numAtoms << " Atoms" << std::endl;

  // -----------------------------------------------------
  // calculate size of grid with a fixed resolution
  // -----------------------------------------------------

  // Find bounding box
  auto [xmin, xmax, ymin, ymax, zmin, zmax] = Coords.ComputeBounds(num_atoms);

  float xlength = xmax - xmin;
  float ylength = ymax - ymin;
  float zlength = zmax - zmin;

  // Compute extents of the molecule for creating the grids
  float scale_model = std::max(std::max(xlength, ylength), zlength);
  scale_model += Coords.LargestVdwRadius() * 2 + r_probe * 2;
  glm::vec3 dim_grid_prelim = glm::vec3(scale_model, scale_model, scale_model);

  const int voxels_per_patch_dim = 64;
  int patches_per_dim;
  try {
    patches_per_dim = std::stoi(argv[2]);
  } catch (const std::invalid_argument& e) {
    std::cerr << "Invalid number: " << argv[2] << std::endl;
    return 1;
  } catch (const std::out_of_range& e) {
    std::cerr << "Number out of range: " << argv[2] << std::endl;
    return 1;
  }
  const int voxels_per_dim = patches_per_dim * voxels_per_patch_dim;
  const float resolution = scale_model / voxels_per_dim;

  //NOTE: optionally set up file to save profiling data
  // std::ofstream outFile("results/deepses_" + filename + "_" + std::to_string(patches_per_dim) + "_patches.txt");
  // bool outfile_written = false;

  // // check that the file opened successfully.
  // if (!outFile) {
  //   std::cerr << "Error: Could not open file for writing\n";
  //   return 1;
  // }

  // // print calculated resolution and number of voxels/patches
  // std::cout << "resolution: " << resolution << " Angstrom / voxel" << std::endl;
  // std::cout << "patches per dim: " << patches_per_dim << std::endl;
  // std::cout << "voxels per dim: " << voxels_per_dim << std::endl;

  // // write to file
  // outFile << "num Atoms: " << Coords.numAtoms << "\n";
  // outFile << "resolution: " << resolution << " Angstrom / voxel\n";
  // outFile << "patches per dim: " << patches_per_dim << "\n";
  // outFile << "voxels per dim: " << voxels_per_dim << std::endl;

  // Grid for storing neighbors info
  float n_resolution = r_probe + Coords.LargestVdwRadius();
  glm::ivec3 n_dim_grid = glm::ivec3(glm::ceil(dim_grid_prelim * (1.0f / n_resolution)));
  
  const int batchDim = patches_per_dim * patches_per_dim * patches_per_dim;
  const glm::ivec3 dim_grid = glm::ivec3(voxels_per_dim, voxels_per_dim, voxels_per_dim);

  // ------------------------------------------------------------------------------
  // Place molecule and camera in global space
  // ------------------------------------------------------------------------------

  //  This is the distance to the center of the molecule in which the camera
  //  is placed
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
  int cell_size = 10;
  int n_grid_size = n_dim_grid.x * n_dim_grid.y * n_dim_grid.z;
  unsigned int grid_counters;
  // Initialize counters and cells to 0
  // dynamic allocation as n_grid_size is only known at runtime
  std::vector<int> counters(n_grid_size, 0);
  std::vector<int> cells(n_grid_size * cell_size, 0);

  glGenBuffers(1, &grid_counters);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid_counters);
  glBufferData(GL_SHADER_STORAGE_BUFFER, n_grid_size * sizeof(int),
               &counters[0], GL_STREAM_COPY); //NOTE: was GL_STREAM_READ
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, grid_counters);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 2);

  unsigned int grid_cells;
  glGenBuffers(1, &grid_cells);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid_cells);
  glBufferData(GL_SHADER_STORAGE_BUFFER,
               n_grid_size * cell_size * sizeof(GLuint), &cells[0],
               GL_STREAM_COPY); //NOTE: was GL_STREAM_READ
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

  checkGLError("bind SSBOs");

  // ------------------------------------------------------------------------------
  // Generate Textures
  // ------------------------------------------------------------------------------

  // //DEBUG:
  // GLint maxImageUnits;
  // glGetIntegerv(GL_MAX_IMAGE_UNITS, &maxImageUnits);
  // printf("Max image bindings: %d\n", maxImageUnits);

  // Textures to hold the information for the final shading
  GLuint tex_pos, tex_normal, tex_color;
  glGenTextures(1, &tex_pos);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, tex_pos);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA,
               GL_FLOAT, NULL);
  glBindImageTexture(1, tex_pos, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);

  glGenTextures(1, &tex_normal);
  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, tex_normal);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA,
               GL_FLOAT, NULL);
  glBindImageTexture(2, tex_normal, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);

  glGenTextures(1, &tex_color);
  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, tex_color);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA,
               GL_FLOAT, NULL);
  glBindImageTexture(3, tex_color, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);

  // 3D Texture for the distance field stores color of the closest atom and
  // distance to surface
  GLuint tex_dist_field;
  glGenTextures(1, &tex_dist_field);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_3D, tex_dist_field);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, dim_grid.x, dim_grid.y, dim_grid.z, 0, GL_RED, GL_FLOAT, NULL);
  glBindImageTexture(0, tex_dist_field, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F);

  checkGLError("create textures");

  // ------------------------------------------------------------------------------
  // shaders
  // ------------------------------------------------------------------------------

  // Calculate the neighborhood grid
  cShader neighbors_shader("shaders/neighbors.cs");
  // Calculate vdw
  cShader phase1_shader("shaders/phase1_vdw.cs");

  // Calculate color, position, and normals
  cShader raymarch_shader("shaders/raymarch-molecule_no_color.cs");
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
  // Set up TensorRT engine
  // ------------------------------------------------------------------------------
  
  // create a CUDA stream to execute
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));

  // //DEBUG for printf
  // size_t current_limit = 0;
  // checkCudaErrors(cudaDeviceGetLimit(&current_limit, cudaLimitPrintfFifoSize));
  // size_t new_limit = 1024 * 1024 * 8; // e.g., 8 MB
  // if (new_limit > current_limit) {
  //     cudaError_t err = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, new_limit);
  //     if (err != cudaSuccess) {
  //         fprintf(stderr, "Warning: Failed to set printf FIFO size: %s\n", cudaGetErrorString(err));
  //         // Continue, but printf might truncate
  //     } else {
  //         printf("Set printf FIFO size to %zu bytes\n", new_limit);
  //     }
  //   }

  // register input and output textures with CUDA
  cudaGraphicsResource *cudaInputResource;
  checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaInputResource, tex_dist_field, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
  cudaArray *cudaInputArray;

  // Define effective dimensions for the reshaped data:
  const int effectiveWidth = voxels_per_patch_dim;          // number of elements per row in each 2D slice
  const int effectiveHeight = voxels_per_patch_dim;         // number of rows per 2D slice
  const int effectiveDepth = batchDim * voxels_per_patch_dim;      // total number of slices (batch * patch depth)

  // Set up dimensions for reshaping with cuda kernels
  const int patchVoxels = voxels_per_patch_dim * voxels_per_patch_dim * voxels_per_patch_dim;
  const int totalPatches = patches_per_dim * patches_per_dim * patches_per_dim;

  // instantiate logger
  class Logger : public nvinfer1::ILogger
  {
      void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
      {
          // suppress info-level messages
          if (severity <= nvinfer1::ILogger::Severity::kWARNING)
              std::cout << msg << std::endl;
      }
  } logger;

  // build TensorRT engine
  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
  // open model from file
  const std::string enginePath = "run/data/engines/unet_4_ch_1-2-4_mults_10_06_25_FP16.trt";

  std::ifstream file(enginePath, std::ios::binary);
  // write to .txt file
  outFile << "engine: " << enginePath << std::endl;

  // set batch size here
  const int batchSize = 64; // for inference

  // Check if the file was opened successfully
  if (!file.good()) {
    std::cerr << "Engine file cannot be opened, please check the file!" << std::endl;
    exit(EXIT_FAILURE);
  }
  // determine size of file
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  // allocate memory to hold file contents
  std::vector<char> engineData(size);
  // read file and close
  file.read(engineData.data(), size);
  file.close();

  // deserialize model to engine
  nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), size);
  if (!engine) {
    std::cerr << "Failed to create engine" << std::endl;
    delete runtime;  // Clean up runtime
  }

  // create execution context
  nvinfer1::IExecutionContext* context = engine->createExecutionContext();
  if (!context) {
    std::cerr << "Failed to create execution context" << std::endl;
    delete engine;  // Clean up engine
  }

  //DEBUG: add a profiler to the execution context
  // MyProfiler myProfiler;
  // context->setProfiler(&myProfiler);

  // Retrieve tensor names dynamically
  std::string inputTensorName, outputTensorName;

  for (int i = 0; i < engine->getNbIOTensors(); ++i)
  {
    const char* tensorName = engine->getIOTensorName(i);

    // Check if it's an input or output tensor
    if (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT)
    {
      inputTensorName = tensorName;
    }
    else if (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kOUTPUT)
    {
      outputTensorName = tensorName;
    }
  }

  // Check if input/output tensors were found
  if (inputTensorName.empty() || outputTensorName.empty())
  {
    std::cerr << "Error: Could not determine input/output tensor names!" << std::endl;
    return -1;
  }

  // set input shapes
  nvinfer1::Dims inputDims;
  inputDims.nbDims = 5;
  inputDims.d[0] = batchSize;  // batch dimension (should match your slice size)
  inputDims.d[1] = 1;   // number of channels
  inputDims.d[2] = voxels_per_patch_dim;  // depth of patch
  inputDims.d[3] = voxels_per_patch_dim;  // height of patch
  inputDims.d[4] = voxels_per_patch_dim;  // width of patch
  context->setInputShape(inputTensorName.c_str(), inputDims);

  // Ensure all dynamic bindings have been defined.
  if (!context->allInputDimensionsSpecified()) {
    auto msg = "Error, not all required dimensions specified.";
    throw std::runtime_error(msg);
  }

  //DEBUG: check tensor shapes
  // nvinfer1::Dims inputdims = engine->getTensorShape(inputTensorName.c_str());
  // std::cout << "Shape: [ " << std::endl;
  // for (int i = 0; i < inputdims.nbDims; ++i)
  // {
  //     std::cout << inputdims.d[i] << " ";
  // }
  // std::cout << "]" << std::endl;

  // nvinfer1::Dims outputdims = engine->getTensorShape(outputTensorName.c_str());
  // std::cout << "Shape: [ " << std::endl;
  // for (int i = 0; i < outputdims.nbDims; ++i)
  // {
  //     std::cout << outputdims.d[i] << " ";
  // }
  // std::cout << "]" << std::endl;

  //DEBUG: inspect network
  // auto inspector = std::unique_ptr<nvinfer1::IEngineInspector>(engine->createEngineInspector());
  // inspector->setExecutionContext(context); // OPTIONAL
  // // std::cout << inspector->getLayerInformation(0, LayerInformationFormat::kJSON); // Print the information of the first layer in the engine.
  // std::cout << inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON); // Print the information of the entire engine.

  //NOTE: set up filtering of patches s.t. only patches in given range are passed through tensorRT engine (for other patches sdf value is the same as for vdw sdf)
  const float sdf_min = 0.0f;
  const float sdf_max = 1.4f;
  const float epsilon = 1e-6f;

  // allocate buffers for filters and filtered Patches
  int *relevantPatches = nullptr;
  int *scanResults = nullptr;  // Prefix sum results
  float *filteredBatchedInputBuffer = nullptr;// Compacted buffer for selected patches
  
  checkCudaErrors(cudaMalloc(&relevantPatches, batchDim * sizeof(int)));
  checkCudaErrors(cudaMalloc(&scanResults, batchDim * sizeof(int)));

  long long compact_buffer_voxels = (long long)batchDim * patchVoxels;
  //NOTE we usually dont need all 8^3 patches to fit in the buffer, so we can save some memory by making it smaller
  #ifdef _WIN32
    if (patches_per_dim == 6) {
      compact_buffer_voxels = (long long)160 * patchVoxels;
    }
    if (patches_per_dim == 8) { //NOTE: can be optimized but is enough for 25 patches per dim with 24 GB VRAM
      compact_buffer_voxels = (long long)350 * patchVoxels;
    }
    if (patches_per_dim == 10) {
      compact_buffer_voxels = (long long)500 * patchVoxels;
    }
    if (patches_per_dim == 12) {
      compact_buffer_voxels = (long long)700 * patchVoxels;
    }
    if (patches_per_dim == 14) {
      compact_buffer_voxels = (long long)800 * patchVoxels;
    }
  #else
    if (patches_per_dim > 8) { //NOTE: can be optimized but is enough for 25 patches per dim with 24 GB VRAM
      compact_buffer_voxels /= 2;
    }
    if (patches_per_dim >= 16) {
      compact_buffer_voxels /= 2;
    }
    if (patches_per_dim >= 20) {
      compact_buffer_voxels /= 4;
      compact_buffer_voxels *= 3;
    }
  #endif
  checkCudaErrors(cudaMalloc(&filteredBatchedInputBuffer, compact_buffer_voxels * sizeof(float)));

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
  phong_shader->setInt("tex_pos", 1);
  phong_shader->setInt("tex_normal", 2);
  phong_shader->setInt("tex_color", 3);
  phong_shader->setVec3("uniform_color", uniform_color);

  // field width from chimerax needs to be halfed
  w *= 0.5;

  // Use high_resolution_clock for potentially better precision
  using Clock = std::chrono::high_resolution_clock;
  using Duration = std::chrono::duration<double>; // Duration in seconds (double)

  static auto last_frame_time = Clock::now();
  static std::vector<double> frame_times; // Optional: for averaging
  static std::vector<int> processed_patches; // Optional: for averaging
  const size_t frame_time_history_size = 36; // Average over 36 frames

  // Initialize timers for performance measurements
  initializeTimers();

  while (!glfwWindowShouldClose(window)) {

    //-------------------------------------------
    // Performance measurements
    //-------------------------------------------

    // --- Calculate indices for timer ring buffer ---
    int writeIndex = currentFrameIndex % NUM_FRAMES_DELAY;
    int readIndex = (currentFrameIndex + 1) % NUM_FRAMES_DELAY; // Read N-2 results (relative to writeIndex)

    // --- 1. Retrieve Results from Frame N - NUM_FRAMES_DELAY ---
    // Avoid retrieving on the very first few frames
    if (currentFrameIndex >= NUM_FRAMES_DELAY) {
        // Retrieve CUDA time
        // Ensure the stop event has actually been recorded by the GPU.
        // cudaEventQuery is non-blocking, but we often need the result now.
        // cudaEventSynchronize blocks the CPU, which is okay here for getting
        // the timing result of a past frame, but avoid it mid-frame.
        cudaError_t cudaSyncStat = cudaEventSynchronize(cudaStopEvents_raymarch[readIndex]);
        if (cudaSyncStat == cudaSuccess) { // Check if synchronization succeeded
             cudaEventElapsedTime(&cudaFrameTimesMs_raymarch[readIndex],
                                  cudaStartEvents_raymarch[readIndex],
                                  cudaStopEvents_raymarch[readIndex]);
        } else {
             // Handle error - event might not have been recorded properly
             cudaFrameTimesMs_raymarch[readIndex] = -1.0f; // Indicate error
             std::cerr << "cuda event for raymarch kernel has not been recorded properly for the frame " << currentFrameIndex - NUM_FRAMES_DELAY << std::endl;
             // Consider logging cudaGetErrorString(cudaSyncStat)
        }
        cudaSyncStat = cudaEventSynchronize(cudaStopEvents_gather[readIndex]);
        if (cudaSyncStat == cudaSuccess) { // Check if synchronization succeeded
             cudaEventElapsedTime(&cudaFrameTimesMs_gather[readIndex],
                                  cudaStartEvents_gather[readIndex],
                                  cudaStopEvents_gather[readIndex]);
        } else {
             // Handle error - event might not have been recorded properly
             cudaFrameTimesMs_gather[readIndex] = -1.0f; // Indicate error
             std::cerr << "cuda event for gather has not been recorded properly for the frame " << currentFrameIndex - NUM_FRAMES_DELAY << std::endl;
             // Consider logging cudaGetErrorString(cudaSyncStat)
        }
        cudaSyncStat = cudaEventSynchronize(cudaStopEvents_inference[readIndex]);
        if (cudaSyncStat == cudaSuccess) { // Check if synchronization succeeded
             cudaEventElapsedTime(&cudaFrameTimesMs_inference[readIndex],
                                  cudaStartEvents_inference[readIndex],
                                  cudaStopEvents_inference[readIndex]);
        } else {
             // Handle error - event might not have been recorded properly
             cudaFrameTimesMs_inference[readIndex] = -1.0f; // Indicate error
             std::cerr << "cuda event for inference has not been recorded properly for the frame " << currentFrameIndex - NUM_FRAMES_DELAY << std::endl;
             // Consider logging cudaGetErrorString(cudaSyncStat)
        }
        cudaSyncStat = cudaEventSynchronize(cudaStopEvents_scatter[readIndex]);
        if (cudaSyncStat == cudaSuccess) { // Check if synchronization succeeded
             cudaEventElapsedTime(&cudaFrameTimesMs_scatter[readIndex],
                                  cudaStartEvents_scatter[readIndex],
                                  cudaStopEvents_scatter[readIndex]);
        } else {
             // Handle error - event might not have been recorded properly
             cudaFrameTimesMs_scatter[readIndex] = -1.0f; // Indicate error
             std::cerr << "cuda event for scatter has not been recorded properly for the frame " << currentFrameIndex - NUM_FRAMES_DELAY << std::endl;
             // Consider logging cudaGetErrorString(cudaSyncStat)
        }
        cudaSyncStat = cudaEventSynchronize(cudaStopEvents_total[readIndex]);
        if (cudaSyncStat == cudaSuccess) { // Check if synchronization succeeded
             cudaEventElapsedTime(&cudaFrameTimesMs_total[readIndex],
                                  cudaStartEvents_total[readIndex],
                                  cudaStopEvents_total[readIndex]);
        } else {
             // Handle error - event might not have been recorded properly
             cudaFrameTimesMs_total[readIndex] = -1.0f; // Indicate error
             std::cerr << "cuda event for total cuda pipeline has not been recorded properly for the frame " << currentFrameIndex - NUM_FRAMES_DELAY << std::endl;
             // Consider logging cudaGetErrorString(cudaSyncStat)
        }


        // Retrieve OpenGL time
        GLint available = 0;
        glGetQueryObjectiv(glQueries[readIndex], GL_QUERY_RESULT_AVAILABLE, &available);

        if (available) {
            glGetQueryObjectui64v(glQueries[readIndex], GL_QUERY_RESULT, &glFrameTimesNs[readIndex]);
        } else {
            // Result not ready yet (should be rare with NUM_FRAMES_DELAY=2+)
            // Keep the old value or mark as invalid
             glFrameTimesNs[readIndex] = 0; // Or some other indicator
            std::cerr << "Warning: GL Query result not ready for frame " << currentFrameIndex - NUM_FRAMES_DELAY << std::endl;
        }

        // --- Optional: Average the valid results over the delay buffer ---
        double validCudaSum_raymarch = 0;
        double validCudaSum_gather = 0;
        double validCudaSum_inference = 0;
        double validCudaSum_scatter = 0;
        double validCudaSum_total = 0;
        int validCudaCount = 0;
        double validGLSum = 0;   int validGLCount = 0;
        for(int i=0; i<NUM_FRAMES_DELAY; ++i) {
            if (cudaFrameTimesMs_raymarch[i] >= 0.0f && cudaFrameTimesMs_gather[i] >= 0.0f && cudaFrameTimesMs_inference[i] >= 0.0f && cudaFrameTimesMs_scatter[i] >= 0.0f && cudaFrameTimesMs_total[i] >= 0.0f) { // Check for valid/non-error time
                validCudaSum_raymarch += cudaFrameTimesMs_raymarch[i];
                validCudaSum_gather += cudaFrameTimesMs_gather[i];
                validCudaSum_inference += cudaFrameTimesMs_inference[i];
                validCudaSum_scatter += cudaFrameTimesMs_scatter[i];
                validCudaSum_total += cudaFrameTimesMs_total[i];
                validCudaCount++;
            }
            if (glFrameTimesNs[i] > 0) { // Check for valid/non-error time
                 validGLSum += (glFrameTimesNs[i] / 1'000'000.0); // Convert ns to ms
                 validGLCount++;
            }
        }
        lastAvgCudaTimeMs_raymarch = (validCudaCount > 0) ? (validCudaSum_raymarch / validCudaCount) : 0.0;
        lastAvgCudaTimeMs_gather = (validCudaCount > 0) ? (validCudaSum_gather / validCudaCount) : 0.0;
        lastAvgCudaTimeMs_inference = (validCudaCount > 0) ? (validCudaSum_inference / validCudaCount) : 0.0;
        lastAvgCudaTimeMs_scatter = (validCudaCount > 0) ? (validCudaSum_scatter / validCudaCount) : 0.0;
        lastAvgCudaTimeMs_total = (validCudaCount > 0) ? (validCudaSum_total / validCudaCount) : 0.0;
        lastAvgGLTimeMs = (validGLCount > 0) ? (validGLSum / validGLCount) : 0.0;

        // Second pass: Calculate sum of squared differences from the mean
        double sumSqDiffCuda_raymarch = 0;
        double sumSqDiffCuda_gather = 0;
        double sumSqDiffCuda_inference = 0;
        double sumSqDiffCuda_scatter = 0;
        double sumSqDiffCuda_total = 0;
        double sumSqDiffGL = 0;

        if (validCudaCount > 0) { // Only if there are valid CUDA samples
            for (int i = 0; i < NUM_FRAMES_DELAY; ++i) {
                if (cudaFrameTimesMs_raymarch[i] >= 0.0f &&
                    cudaFrameTimesMs_gather[i] >= 0.0f &&
                    cudaFrameTimesMs_inference[i] >= 0.0f &&
                    cudaFrameTimesMs_scatter[i] >= 0.0f &&
                    cudaFrameTimesMs_total[i] >= 0.0f) {

                    sumSqDiffCuda_raymarch += std::pow(cudaFrameTimesMs_raymarch[i] - lastAvgCudaTimeMs_raymarch, 2);
                    sumSqDiffCuda_gather += std::pow(cudaFrameTimesMs_gather[i] - lastAvgCudaTimeMs_gather, 2);
                    sumSqDiffCuda_inference += std::pow(cudaFrameTimesMs_inference[i] - lastAvgCudaTimeMs_inference, 2);
                    sumSqDiffCuda_scatter += std::pow(cudaFrameTimesMs_scatter[i] - lastAvgCudaTimeMs_scatter, 2);
                    sumSqDiffCuda_total += std::pow(cudaFrameTimesMs_total[i] - lastAvgCudaTimeMs_total, 2);
                }
            }
        }

        if (validGLCount > 0) { // Only if there are valid GL samples
            for (int i = 0; i < NUM_FRAMES_DELAY; ++i) {
                if (glFrameTimesNs[i] > 0) {
                    double glTimeMs = glFrameTimesNs[i] / 1'000'000.0;
                    sumSqDiffGL += std::pow(glTimeMs - lastAvgGLTimeMs, 2);
                }
            }
        }

        // Calculate Standard Deviations
        // (Population Standard Deviation: divide by N. Sample Standard Deviation: divide by N-1)
        // We'll use Population Standard Deviation here (divide by N) as it's simpler and common for this kind of running average.
        // If validCudaCount is 1, std dev is 0.
        stdDevCudaTimeMs_raymarch = (validCudaCount > 0) ? std::sqrt(sumSqDiffCuda_raymarch / validCudaCount) : 0.0;
        stdDevCudaTimeMs_gather = (validCudaCount > 0) ? std::sqrt(sumSqDiffCuda_gather / validCudaCount) : 0.0;
        stdDevCudaTimeMs_inference = (validCudaCount > 0) ? std::sqrt(sumSqDiffCuda_inference / validCudaCount) : 0.0;
        stdDevCudaTimeMs_scatter = (validCudaCount > 0) ? std::sqrt(sumSqDiffCuda_scatter / validCudaCount) : 0.0;
        stdDevCudaTimeMs_total = (validCudaCount > 0) ? std::sqrt(sumSqDiffCuda_total / validCudaCount) : 0.0;

        stdDevGLTimeMs = (validGLCount > 0) ? std::sqrt(sumSqDiffGL / validGLCount) : 0.0;

        // Now you can display lastAvgCudaTimeMs and lastAvgGLTimeMs
        std::cout << "Avg CUDA times:" << std::endl;
        std::cout << "Raymarch: " << lastAvgCudaTimeMs_raymarch << " +/- " << stdDevCudaTimeMs_raymarch << " ms" << std::endl;
        std::cout << "Gather: " << lastAvgCudaTimeMs_gather << " +/- " << stdDevCudaTimeMs_gather << " ms" << std::endl;
        std::cout << "Inference: " << lastAvgCudaTimeMs_inference << " +/- " << stdDevCudaTimeMs_inference << " ms" << std::endl;
        std::cout << "Scatter: " << lastAvgCudaTimeMs_scatter << " +/- " << stdDevCudaTimeMs_scatter << " ms" << std::endl;
        std::cout << "------------------------" << std::endl; 
        std::cout << "Total CUDA time: " << lastAvgCudaTimeMs_total << " +/- " << stdDevCudaTimeMs_total << " ms" << std::endl;
        std::cout << "Avg GL time: " << lastAvgGLTimeMs << " +/- " << stdDevGLTimeMs << " ms" << std::endl;

        //NOTE optionally print to file
        // if ((currentFrameIndex >= 2 * NUM_FRAMES_DELAY) && !outfile_written) {
        //   outFile << "Avg CUDA times:" << std::endl;
        //   outFile << "Raymarch: " << lastAvgCudaTimeMs_raymarch << " +/- " << stdDevCudaTimeMs_raymarch << " ms" << std::endl;
        //   outFile << "Gather: " << lastAvgCudaTimeMs_gather << " +/- " << stdDevCudaTimeMs_gather << " ms" << std::endl;
        //   outFile << "Inference: " << lastAvgCudaTimeMs_inference << " +/- " << stdDevCudaTimeMs_inference << " ms" << std::endl;
        //   outFile << "Scatter: " << lastAvgCudaTimeMs_scatter << " +/- " << stdDevCudaTimeMs_scatter << " ms" << std::endl;
        //   outFile << "------------------------" << std::endl; 
        //   outFile << "Total CUDA time: " << lastAvgCudaTimeMs_total << " +/- " << stdDevCudaTimeMs_total << " ms" << std::endl;
        //   outFile << "Avg GL time: " << lastAvgGLTimeMs << " +/- " << stdDevGLTimeMs << " ms" << std::endl;
        // }
    } else if (currentFrameIndex == 1) {
      std::cout << "Collecting frames for averaging..." << std::endl;
    }

    // Measure FPS
    auto current_frame_time = Clock::now();
    Duration delta_time_duration = current_frame_time - last_frame_time;
    double delta_time_sec = delta_time_duration.count();
    last_frame_time = current_frame_time;

    // --- Optional: Averaging for smoother FPS display ---
    frame_times.push_back(delta_time_sec);
    if (frame_times.size() > frame_time_history_size) {
        frame_times.erase(frame_times.begin()); // Remove oldest
    }

    double average_delta_time = 0.0;
    double std_dev_delta_time = 0.0;

    if (!frame_times.empty()) {
        double sum = std::accumulate(frame_times.begin(), frame_times.end(), 0.0);
        average_delta_time = sum / frame_times.size();

        // Calculate Standard Deviation for delta_time_sec
        if (frame_times.size() > 0) { // StdDev only meaningful for >0 elements
          double sum_sq_diff_delta_time = 0.0;
          for (double dt : frame_times) {
              sum_sq_diff_delta_time += std::pow(dt - average_delta_time, 2);
          }
          double variance_delta_time = sum_sq_diff_delta_time / frame_times.size(); // Population variance
          std_dev_delta_time = std::sqrt(variance_delta_time);
        }
    }
    double average_fps = (average_delta_time > 0.0) ? (1.0 / average_delta_time) : 0.0;
    // --- End Optional Averaging ---

    double instantaneous_fps = (delta_time_sec > 0.0) ? (1.0 / delta_time_sec) : 0.0;

    // Display FPS
    std::cout << "FPS: " << instantaneous_fps << " , avg. FPS: " << average_fps << std::endl;
    std::cout << "avg. frame time: " << average_delta_time * 1000.0 << " +/- " << std_dev_delta_time * 1000.0 << "ms" << std::endl;

    //NOTE: optionally print to file
    // if ((currentFrameIndex >= 2 * NUM_FRAMES_DELAY) && !outfile_written) {
    //   outFile << "FPS: " << instantaneous_fps << " , avg. FPS: " << average_fps << std::endl;
    //   outFile << "avg. frame time: " << average_delta_time * 1000.0 << " +/- " << std_dev_delta_time * 1000.0 << "ms" << std::endl;
    // }

    // --- avg processed patches ---
    if (processed_patches.size() > frame_time_history_size) {
        processed_patches.erase(processed_patches.begin()); // Remove oldest
    }

    double average_processed_patches = 0.0;
    double std_dev_processed_patches = 0.0;

    if (!processed_patches.empty()) {
        double sum = std::accumulate(processed_patches.begin(), processed_patches.end(), 0.0);
        average_processed_patches = sum / processed_patches.size();
        // Calculate Standard Deviation for processed_patches
        if (processed_patches.size() > 0) { // StdDev only meaningful for >0 elements
          double sum_sq_diff_patches = 0.0;
          for (auto patch_count : processed_patches) { // Use auto to handle int or double
              sum_sq_diff_patches += std::pow(static_cast<double>(patch_count) - average_processed_patches, 2);
          }
          double variance_patches = sum_sq_diff_patches / processed_patches.size(); // Population variance
          std_dev_processed_patches = std::sqrt(variance_patches);
        }
    }

    std::cout << "avg. processed patches: " << average_processed_patches << " +/- " << std_dev_processed_patches << std::endl;
    //NOTE: optionally print to file + also write gpu memory usage to file
    // if ((currentFrameIndex >= 2 * NUM_FRAMES_DELAY) && !outfile_written) {
    //   outFile << "avg. processed patches: " << average_processed_patches << " +/- " << std_dev_processed_patches << std::endl;
    //   // print gpu memory during render loop
    //   // printNvmlMem("mid-frame");

    //   // determine gpu memory mid-frame
    //   nvmlInit();
    //   nvmlDevice_t dev;
    //   nvmlDeviceGetHandleByIndex(0, &dev);  // GPU 0
    //   nvmlMemory_t memInfo;
    //   nvmlDeviceGetMemoryInfo(dev, &memInfo);
    //   float gpu_memory = memInfo.used / (1024.0*1024.0) - gpu_memory_start;
    //   outFile << "GPU memory used: " << gpu_memory << " MB" << std::endl;
    //   nvmlShutdown();

    //   // write screen size to file
    //   outFile << "screen size: " << SCR_WIDTH << " x " << SCR_HEIGHT << std::endl;

    //   outfile_written = true;
    //   break; //NOTE end render loop
      
    // }
    // --- end avg processed patches ---

    // ------------------------------------------------------------------------------
    // Camera
    // ------------------------------------------------------------------------------

    //NOTE: Uncomment for rotating molecule
    camera.rotate_angle_axis(glm::radians(10.0), glm::vec3(0, 1, 0));

    camera_changed = true; //NOTE: do not change!

    if (!setView) {
      view = camera.GetViewMatrix();
      camera_pos = camera.getPosition();
      w = camera.getRadius() * 0.5f;
      camera_front = glm::normalize(camera.getFront());
    }
    float aspect = (float)SCR_WIDTH / SCR_HEIGHT;
    glm::mat4 projection =
        glm::ortho(-w, w, -w / aspect, w / aspect, -200.0f, 200.0f);

    if (camera_changed) {
      if (model_changed) {
        // ------------------------------------------------------------------------------
        // Compute neighbors grid
        // ------------------------------------------------------------------------------

        // Begin GL timer query BEFORE the OpenGL commands you want to measure
        glBeginQuery(GL_TIME_ELAPSED, glQueries[writeIndex]);

        neighbors_shader.use();
        checkGLError("use neighbors_shader");

        glDispatchCompute(num_atoms, 1, 1);
        checkGLError("dispatch compute neighbors_shader");

        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        checkGLError("memory barrier neighbors_shader");
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, grid_counters);

        // ------------------------------------------------------------------------------
        // Phase 1: Probe intersection
        // ------------------------------------------------------------------------------
        phase1_shader.use();
        checkGLError("use phase1_shader");

        glDispatchCompute(dim_grid.x / 8, dim_grid.y / 8, dim_grid.z / 8);
        checkGLError("dispatch compute phase1_shader");

        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        checkGLError("memory barrier phase1_shader");

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
        checkGLError("clear neighbors SSBOs");

        // End GL timer query AFTER the OpenGL commands
        glEndQuery(GL_TIME_ELAPSED);

        // ##############################################################################
        // ------------------------------------------------------------------------------
        // START: Compute SES with engine
        // ------------------------------------------------------------------------------
        // ##############################################################################
        
        // Record START event for CUDA operations
        checkCudaErrors(cudaEventRecord(cudaStartEvents_total[writeIndex], stream));

        // map input texture into CUDA
        cudaGraphicsMapResources(1, &cudaInputResource, stream);
        checkCudaErrors(cudaGetLastError());
        cudaGraphicsSubResourceGetMappedArray(&cudaInputArray, cudaInputResource, 0, 0);
        checkCudaErrors(cudaGetLastError());
        
        // Check if array is valid (important!)
        if (cudaInputArray == nullptr) {
          // Handle error: mapping or getting array failed
          cudaGraphicsUnmapResources(1, &cudaInputResource, stream); // Unmap before exiting/throwing
          throw std::runtime_error("Failed to get mapped CUDA array from GL texture");
          return -1; // Or other error handling
        }

        // Create Surface Object for writing to cudaOutputArray
        cudaSurfaceObject_t d_outputSurf = 0;
        cudaResourceDesc resDescSurf;
        memset(&resDescSurf, 0, sizeof(resDescSurf));
        resDescSurf.resType = cudaResourceTypeArray;
        resDescSurf.res.array.array = cudaInputArray;

        checkCudaErrors(cudaCreateSurfaceObject(&d_outputSurf, &resDescSurf));

        // ------------------------------------------------------------------------------
        // find relevant patches with CUDA raymarcher
        // relevant = a voxel is visible and in relevant range
        // ------------------------------------------------------------------------------
        
        // --- Setup CUDA Texture Object ---
        cudaTextureObject_t d_gridTextureObj = 0;
        cudaResourceDesc texResDesc;
        memset(&texResDesc, 0, sizeof(texResDesc));
        texResDesc.resType = cudaResourceTypeArray;
        texResDesc.res.array.array = cudaInputArray;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp; // Or Border for outside values
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.addressMode[2] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear; // Use linear interpolation for sampling distance
        texDesc.readMode = cudaReadModeElementType; // Read as float
        texDesc.normalizedCoords = 1; // Use normalized coordinates [0, 1]

        checkCudaErrors(cudaCreateTextureObject(&d_gridTextureObj, &texResDesc, &texDesc, NULL));

        // --- Setup Kernel Parameters ---
        int3 gridDimsVoxels = {voxels_per_dim, voxels_per_dim, voxels_per_dim};
        int3 patchDimsVoxels = {voxels_per_patch_dim, voxels_per_patch_dim, voxels_per_patch_dim};
        int3 patches_per_dim_int3 = {patches_per_dim, patches_per_dim, patches_per_dim};

        RaymarchParams raymarchParams;
        raymarchParams.gridTexture = d_gridTextureObj;
        raymarchParams.gridDimsVoxels = gridDimsVoxels;
        raymarchParams.gridRes = resolution;
        raymarchParams.patchDimsVoxels = patchDimsVoxels;
        raymarchParams.patchesPerDim = patches_per_dim_int3;
        raymarchParams.totalPatches = batchDim;
        raymarchParams.cameraPos = make_float3(camera_pos.x, camera_pos.y, camera_pos.z);
        raymarchParams.cameraFront = make_float3(camera_front.x, camera_front.y, camera_front.z);

        // Calculate Inverse View-Projection Matrix (using GLM for example)
        glm::mat4 invVP = glm::inverse(projection * view);
        glm::mat4 transposed_invVP = glm::transpose(invVP); //GLM is column-major, but float4x4 and mat_vec_mul needs row-major
        memcpy(&raymarchParams.invViewProj, glm::value_ptr(transposed_invVP), sizeof(float4x4));

        raymarchParams.resolution = make_uint2(SCR_WIDTH, SCR_HEIGHT);
        raymarchParams.epsilon = 0.001f;
        raymarchParams.maxSteps = 2000;
        raymarchParams.visitedPatches = relevantPatches;

        // --- Launch Kernel ---

        dim3 threadsRaymarchKernel(8, 8); // Example block size //NOTE: 8x8 or 16x16 threads seems good
        dim3 blocksRaymarchKernel((SCR_WIDTH / 8 + threadsRaymarchKernel.x - 1) / threadsRaymarchKernel.x, (SCR_HEIGHT / 8 + threadsRaymarchKernel.y - 1) / threadsRaymarchKernel.y); //NOTE take only every second/4th pixel or so

        // for only CheckRangeKernel
        dim3 threadsCheckRangeKernel(512); //NOTE: tune to your gpu
        dim3 blocksCheckRangeKernel(batchDim); //NOTE needs to be the number of patches as is used for indexing

        // Record START event for CUDA operations
        checkCudaErrors(cudaEventRecord(cudaStartEvents_raymarch[writeIndex], stream));

        launchRaymarchAndCheckRangeKernel(raymarchParams, sdf_min, sdf_max, epsilon, blocksRaymarchKernel, threadsRaymarchKernel, stream);
        // launchCheckRangeKernel(raymarchParams, sdf_min, sdf_max, epsilon, blocksCheckRangeKernel, threadsCheckRangeKernel, stream); //NOTE: no raymarching filter

        // NOTE: can use this instead to set all patches to 1 (relevant)
        // dim3 blocks(patches_per_dim);
        // dim3 threads(patches_per_dim * patches_per_dim);
        // launchInitializeFlagsToInt(relevantPatches, batchDim, 1, blocks, threads, stream);
        
        checkCudaErrorCode(cudaGetLastError());

        // Record STOP event for CUDA operations
        checkCudaErrors(cudaEventRecord(cudaStopEvents_raymarch[writeIndex], stream));

        //DEBUG
        // checkCudaErrors(cudaDeviceSynchronize());
        // std::cout << "relevant patches after AndKernel: " << std::endl;
        // visualizeCudaBuffer(relevantPatches, 8, 8, 8, stream);

        // --- Prefix Sum (Scan) using Thrust ---

        // perform scan with kernel
        int selected_count = perform_thrust_scan_and_reduce(
            relevantPatches, // Pass the raw device pointer
            scanResults,     // Pass the raw device pointer
            batchDim,        // Pass the size
            stream           // Pass the stream
        );
        checkCudaErrors(cudaGetLastError());

        // // DEBUG:
        // checkCudaErrors(cudaDeviceSynchronize());
        // std::cout << "relevant patches after scan: " << std::endl;
        // visualizeCudaBuffer(relevantPatches, 8, 8, 8, stream);

        // std::cout << "thrust scan results: " << std::endl;
        // visualizeCudaBuffer(scanResults, 8, 8, 8, stream);
        // checkCudaErrors(cudaDeviceSynchronize());

        // std::cout << "Found " << selected_count << " patches." << std::endl;

        // save number of processed patches for averaging
        processed_patches.push_back(selected_count);

        // ------------------------------------------------------------------------------
        // Gather Selected Patches into filteredBatchedInputBuffer
        // ------------------------------------------------------------------------------

        dim3 threadsPerBlock(512); //NOTE: tune for you gpu

        if (selected_count == 0) {
            std::cout << "No patches selected. Skipping gather, processing, and scatter." << std::endl;
        } else {
          dim3 blocksForGather(batchDim); // Iterate over original patches

          // Record START event for CUDA operations
          checkCudaErrors(cudaEventRecord(cudaStartEvents_gather[writeIndex], stream));

          launchGatherPatchesFromTexture(
            d_gridTextureObj, filteredBatchedInputBuffer,
            relevantPatches, scanResults, 
            voxels_per_patch_dim,
            patches_per_dim,
            batchDim, 
            blocksForGather, threadsPerBlock, stream
          );
          checkCudaErrors(cudaGetLastError());

          // Record STOP event for CUDA operations
          checkCudaErrors(cudaEventRecord(cudaStopEvents_gather[writeIndex], stream));
        }

        // ------------------------------------------------------------------------------
        // Run Inference on selected Patches
        // ------------------------------------------------------------------------------

        // Record START event for CUDA operations
        checkCudaErrors(cudaEventRecord(cudaStartEvents_inference[writeIndex], stream));

        for (int batch = 0; batch < (selected_count + batchSize - 1) / batchSize; batch++) { // ceil division

          // Compute pointer offsets for this batch.
          size_t offset = batch * batchSize * patchVoxels;

          float* batchInput  = reinterpret_cast<float*>(filteredBatchedInputBuffer) + offset;
          float* batchOutput = reinterpret_cast<float*>(filteredBatchedInputBuffer) + offset; //TODO can i remove this?
          
          // set batch size in tensor shape for last inference call if neccessary
          int actualBatchSize = batchSize;
          nvinfer1::Dims lastInputDims;
          if ((batch + 1) * batchSize > selected_count) {
            actualBatchSize = selected_count - batch * batchSize;
          }
          assert(offset + actualBatchSize * patchVoxels <= dim_grid.x * dim_grid.y * dim_grid.z);

          lastInputDims.nbDims = 5;
          lastInputDims.d[0] = actualBatchSize;  // batch dimension
          lastInputDims.d[1] = 1;   // number of channels
          lastInputDims.d[2] = voxels_per_patch_dim;  // depth of patch
          lastInputDims.d[3] = voxels_per_patch_dim;  // height of patch
          lastInputDims.d[4] = voxels_per_patch_dim;  // width of patch
          context->setInputShape(inputTensorName.c_str(), lastInputDims);
          
          // Bind the input and output buffers by tensor name.
          if (!context->setTensorAddress(inputTensorName.c_str(), static_cast<void*>(batchInput))) {
            throw std::runtime_error("Adress to input tensor not set correctly.");
          }
          if (!context->setTensorAddress(outputTensorName.c_str(), static_cast<void*>(batchOutput))) {
            throw std::runtime_error("Adress to output tensor not set correctly.");
          }

          if (reinterpret_cast<uintptr_t>(batchInput) % 256 != 0 || reinterpret_cast<uintptr_t>(batchOutput) % 256 != 0) {
            std::cerr << "Warning: TensorRT tensor address is not 256-byte aligned!" << std::endl;
          }

          // Enqueue inference asynchronously on the given CUDA stream.
          if (!context->enqueueV3(stream)) {
            std::cerr << "Error during inference on batch " << batch << std::endl;
            // Handle error appropriately.
          }

        }

        // Record STOP event for CUDA operations
        checkCudaErrors(cudaEventRecord(cudaStopEvents_inference[writeIndex], stream));

        // //DEBUG: print inference timings
        // myProfiler.printLayerTimes();
        // myProfiler.reset();

        // ------------------------------------------------------------------------------
        // Scatter Processed and Unprocessed Patches to OutputArray
        // ------------------------------------------------------------------------------

        dim3 blocksForScatter(batchDim); // Iterate over original patches

        // Record START event for CUDA operations
        checkCudaErrors(cudaEventRecord(cudaStartEvents_scatter[writeIndex], stream));

        launchScatterPatchesToSurfaceKernel(
            filteredBatchedInputBuffer,       // Source: processed data
            d_gridTextureObj,                  // Source: unprocessed data
            d_outputSurf,                      // Destination
            relevantPatches, scanResults,
            voxels_per_patch_dim, patches_per_dim, batchDim,
            blocksForScatter, threadsPerBlock, stream
        );
        checkCudaErrors(cudaGetLastError());

        // Record STOP event for CUDA operations
        checkCudaErrors(cudaEventRecord(cudaStopEvents_scatter[writeIndex], stream));

        // clear scan results buffer
        checkCudaErrors(cudaMemsetAsync(scanResults, 0, batchDim * sizeof(int), stream));

        // --- Clear relevant patches Buffer ---
        checkCudaErrors(cudaMemsetAsync(relevantPatches, 0, batchDim * sizeof(int), stream));

        // --- Cleanup ---
        // Destroy texture object when done with it (e.g., end of program/context)
        checkCudaErrors(cudaDestroyTextureObject(d_gridTextureObj));
        checkCudaErrors(cudaDestroySurfaceObject(d_outputSurf));

        // unmap the input and output resources
        cudaGraphicsUnmapResources(1, &cudaInputResource, stream);
        // cudaGraphicsUnmapResources(1, &cudaOutputResource, stream);

        // Record STOP event for CUDA operations
        checkCudaErrors(cudaEventRecord(cudaStopEvents_total[writeIndex], stream));

        // bind the output texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, tex_dist_field);

        model_changed = true; //NOTE: if set to 'false' do not recompute sdf for next frame
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

    // Increment Frame Counter for performance measurements
    currentFrameIndex++;
  }

  // Clean up
  glDeleteVertexArrays(1, &quadVAO);
  glDeleteBuffers(1, &quadVBO);

  delete raymarch_shader;
  delete phong_shader;

  // Clean up timers for performance measurements
  cleanupTimers();

  // Clean up cuda stuff
  checkCudaErrors(cudaFree(filteredBatchedInputBuffer));
  checkCudaErrors(cudaFree(relevantPatches));
  checkCudaErrors(cudaFree(scanResults));
  cudaStreamDestroy(stream);

  outFile.close();

  glfwTerminate();
  return 0;
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
  SCR_HEIGHT = height;
  SCR_WIDTH = width;
  camera.setScreenSize(width, height);
  glActiveTexture(GL_TEXTURE1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA,
               GL_FLOAT, NULL);
  glActiveTexture(GL_TEXTURE2);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA,
               GL_FLOAT, NULL);
  glActiveTexture(GL_TEXTURE3);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA,
               GL_FLOAT, NULL);
  glActiveTexture(GL_TEXTURE0);
  camera_changed = true;
}

// rotate the object with the mouse using and arcball transformation
void mouse_callback(GLFWwindow *window, double xpos, double ypos) {
  // rotate the object with the mouse using and arcball transformation
  int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
  if (state == GLFW_PRESS) {
    camera.rotate(xpos, ypos);
  }
  camera_changed = true;
  camera.setPos(xpos, ypos);
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