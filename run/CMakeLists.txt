
# --------------------------------------
# WINDOWS

# NOTE: To run this under Windows one needs to install the needed dependencies by hand which are handled by the singularity container under linux. Also one needs to adapt the search paths for the dependencies manually.
# --------------------------------------
if(WIN32)

    cmake_minimum_required(VERSION 3.18) # test for versions below 3.18

    # Print CMake's view of the PATH
    message(STATUS "CMake PATH: $ENV{PATH}")

    set(CMAKE_INCLUDE_CURRENT_DIR ON)
    project(
            uncertainvis
            LANGUAGES CXX C CUDA)

    # set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # Specify that CUDA should use C++11 (or higher) features:
    set(CMAKE_CUDA_ARCHITECTURES 86) # Change 86 based on your GPU architectures compute capability
    # set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # --- Windows Specific ---
    if(WIN32)
      # Add definitions needed on Windows (e.g., suppress warnings, specify API versions)
      add_definitions(-D_USE_MATH_DEFINES -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS)
      # Set MSVC specific flags if needed (e.g., /MP for parallel builds)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    endif()

    include(FetchContent)

    set(${SHADER_DIR} "${CMAKE_SOURCE_DIR}/shaders/")

    # Include glfw
    FetchContent_Declare(
            glfw
            GIT_REPOSITORY https://github.com/glfw/glfw
    )
    FetchContent_MakeAvailable(glfw)

    # set(FETCH_ZLIB_NG "ON")
    # Find ZLIB (vcpkg integration should handle this)
    # The HINTS might not even be needed if integration worked
    find_package(ZLIB REQUIRED)

    if(NOT ZLIB_FOUND)
        message(FATAL_ERROR "ZLIB not found. Ensure vcpkg integration is active (`vcpkg integrate install`) or specify vcpkg toolchain file.")
    else()
        message(STATUS "ZLIB Found via vcpkg (presumably): TRUE")
        # Variables like ZLIB_INCLUDE_DIRS and ZLIB_LIBRARIES should be populated correctly
        message(STATUS "ZLIB Include Dirs: ${ZLIB_INCLUDE_DIRS}")
        message(STATUS "ZLIB Libraries: ${ZLIB_LIBRARIES}")
        # If using modern CMake targets (vcpkg usually provides them):
        if(TARGET ZLIB::ZLIB)
             message(STATUS "ZLIB Target: ZLIB::ZLIB")
        endif()
    endif()

    # include gemmi

    # Find CUDA
    find_package(CUDA REQUIRED)
    message(STATUS "CUDA Found: ${CUDA_FOUND}")
    message(STATUS "CUDA Toolkit Root: ${CUDA_TOOLKIT_ROOT_DIR}")
    # if(TARGET CUDA::cudart)
    #     message(STATUS "CUDA::cudart target FOUND.")
    # else()
    #     message(WARNING "CUDA::cudart target WAS NOT FOUND by find_package!")
    # endif()
    # find_package(CUDAToolkit REQUIRED)
    # message(STATUS ">>> CUDAToolkit_FOUND = ${CUDAToolkit_FOUND}")
    # message(STATUS ">>> CUDAToolkit includes = ${CUDAToolkit_INCLUDE_DIRS}")
    # # Check if the specific target exists right after find_package
    # if(TARGET CUDA::cudart)
    #     message(STATUS ">>> Target CUDA::cudart FOUND immediately after find_package.")
    # else()
    #     message(WARNING ">>> Target CUDA::cudart NOT FOUND immediately after find_package!")
    # endif()

     # Look for the NVML library in CUDA’s lib64 plus system paths
    find_library(NVML_LIBRARY
        NAMES nvml nvidia-ml
        HINTS
            ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
    )

    if (NOT NVML_LIBRARY)
        message(FATAL_ERROR "Could not find NVML library (libnvidia-ml.so / nvml.dll)")
    endif()
        
    # --- Find TensorRT ---
    set(TensorRT_SEARCH_PATHS
        "C:/Program Files/TensorRT-10.9.0.34/TensorRT-10.9.0.34"
    )

    find_path(TensorRT_INCLUDE_DIR NvInfer.h
        REQUIRED
        HINTS ${TensorRT_SEARCH_PATHS}
        PATH_SUFFIXES include
        DOC "Path to TensorRT include directory"
    )

    find_library(TensorRT_NVINFER_LIB nvinfer_10.lib
        REQUIRED
        HINTS ${TensorRT_SEARCH_PATHS}
        PATH_SUFFIXES lib lib/x64 # Common suffixes
        DOC "Path to TensorRT nvinfer library"
    )

    # Find plugin library (often needed)
    find_library(TensorRT_PLUGIN_LIBRARY nvinfer_plugin_10.lib
        REQUIRED
        HINTS ${TensorRT_SEARCH_PATHS}
        PATH_SUFFIXES lib lib/x64
        DOC "Path to TensorRT nvinfer_plugin library"
    )

    # # Find parsers (if needed)
    # find_library(TensorRT_NVPARSERS_LIBRARY nvparsers
    #      REQUIRED
    #      HINTS ${TensorRT_SEARCH_PATHS}
    #      PATH_SUFFIXES lib lib/x64
    #      DOC "Path to TensorRT nvparsers library"
    # )

    find_library(TensorRT_NVONNXPARSER_LIBRARY nvonnxparser_10.lib
        REQUIRED
        HINTS ${TensorRT_SEARCH_PATHS}
        PATH_SUFFIXES lib lib/x64
        DOC "Path to TensorRT nvonnxparser library"
    )

    # --- Create a list of found libraries ---
    set(TensorRT_LIBRARIES "") # Initialize empty list

    if(TensorRT_NVINFER_LIB)
        list(APPEND TensorRT_LIBRARIES ${TensorRT_NVINFER_LIB})
    endif()
    if(TensorRT_PLUGIN_LIB)
        list(APPEND TensorRT_LIBRARIES ${TensorRT_PLUGIN_LIB})
    endif()
    # if(TensorRT_NVPARSERS_LIB)
    #     list(APPEND TensorRT_LIBRARIES ${TensorRT_NVPARSERS_LIB})
    # endif()
    if(TensorRT_NVONNXPARSER_LIB)
        list(APPEND TensorRT_LIBRARIES ${TensorRT_NVONNXPARSER_LIB})
    endif()


    if(NOT TensorRT_INCLUDE_DIR OR NOT TensorRT_NVINFER_LIB)
        message(FATAL_ERROR "TensorRT include directory or nvinfer library not found. "
                            "Adjust TensorRT_SEARCH_PATHS in CMakeLists.txt. "
                            " Searched paths: ${TensorRT_SEARCH_PATHS}")
    else()
        message(STATUS "TensorRT Include Dir: ${TensorRT_INCLUDE_DIR}")
        message(STATUS "TensorRT nvinfer Library: ${TensorRT_NVINFER_LIB}")
        if(TensorRT_PLUGIN_LIBRARY)
             message(STATUS "TensorRT Plugin Library: ${TensorRT_PLUGIN_LIBRARY}")
        endif()
         # if(TensorRT_NVPARSERS_LIBRARY)
         #     message(STATUS "TensorRT Parsers Library: ${TensorRT_NVPARSERS_LIBRARY}")
         # endif()
         if(TensorRT_NVONNXPARSER_LIBRARY)
             message(STATUS "TensorRT ONNX Parser Library: ${TensorRT_NVONNXPARSER_LIBRARY}")
         endif()

        # Add include directory globally or per-target
        # include_directories(${TensorRT_INCLUDE_DIR}) # Global (simpler for one project)
    endif()
    # --- End Find TensorRT ---

    set(ONLY_PROGRAM ON) # to not build python module
    FetchContent_Declare(
      gemmi
      GIT_REPOSITORY https://github.com/project-gemmi/gemmi.git
      GIT_TAG        ...
    )
    FetchContent_GetProperties(gemmi)
    if (NOT gemmi_POPULATED)
      FetchContent_Populate(gemmi)
      add_subdirectory(${gemmi_SOURCE_DIR} ${gemmi_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
    FetchContent_MakeAvailable(gemmi)

    # find python path NOTE: needs to be ATER gemmi
    # Force CMake to use a specific Python executable
    # Replace with the *exact* path to your python.exe
    set(Python_EXECUTABLE "C:/Users/niklas/AppData/Local/Programs/Python/Python312/python.exe" CACHE FILEPATH "Path to Python executable")

    # find_program(PYTHON_EXE_PATH python DOC "Python interpreter" REQUIRED)
    # set(Python_EXECUTABLE ${PYTHON_EXE_PATH} CACHE FILEPATH "Path to Python executable")

    # Verify it's set
    if(DEFINED Python_EXECUTABLE AND EXISTS ${Python_EXECUTABLE})
        message(STATUS "Using Python Executable: ${Python_EXECUTABLE}")
    else()
        message(WARNING "Python_EXECUTABLE variable not set or points to non-existent file: ${Python_EXECUTABLE}")
    endif()

    # Include glm
    FetchContent_Declare(
            glm
            GIT_REPOSITORY https://github.com/g-truc/glm.git
    )
    FetchContent_MakeAvailable(glm)

    # Include glad
    FetchContent_Declare(
            glad
            GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    )
    FetchContent_MakeAvailable(glad)

    # # Include lygia
    # FetchContent_Declare(
    #         lygia
    #         GIT_REPOSITORY https://github.com/patriciogonzalezvivo/lygia.git
    #         GIT_TAG "origin/main"
    # )
    # FetchContent_MakeAvailable(lygia)

    # # Include assimp
    # FetchContent_Declare(
    # assimp
    # GIT_REPOSITORY https://github.com/assimp/assimp.git
    # )
    # FetchContent_MakeAvailable(assimp)

    # --- OpenGL ---
    find_package(OpenGL REQUIRED) # Finds opengl32.lib on Window

    # The executable code is here
    add_subdirectory(apps)

else()

    # ------------------------------------------------------------
    # LINUX
    # ------------------------------------------------------------

    cmake_minimum_required(VERSION 3.14...3.18) # test for versions below 3.18

    # set(CMAKE_INCLUDE_CURRENT_DIR ON)
    project(
            uncertainvis
            LANGUAGES CXX C CUDA)

    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # Specify that CUDA should use C++11 (or higher) features:
    set(CMAKE_CUDA_ARCHITECTURES 86) #NOTE Change 86 based on your GPU architectures compute capability
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    include(FetchContent)

    set(${SHADER_DIR} "${CMAKE_SOURCE_DIR}/shaders/")

    # Include glfw
    FetchContent_Declare(
            glfw
            GIT_REPOSITORY https://github.com/glfw/glfw
    )
    FetchContent_MakeAvailable(glfw)

    # include gemmi

    # set(FETCH_ZLIB_NG "ON")
    find_package(ZLIB REQUIRED)

    # Find CUDA
    find_package(CUDA REQUIRED)
    message(STATUS "CUDA Found: ${CUDA_FOUND}")
    message(STATUS "CUDA Toolkit Root: ${CUDA_TOOLKIT_ROOT_DIR}")

    # Look for the NVML library in CUDA’s lib64 plus system paths
    find_library(NVML_LIBRARY
        NAMES nvml nvidia-ml
        HINTS
            ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs
    )

    if (NOT NVML_LIBRARY)
        message(FATAL_ERROR "Could not find NVML library (libnvidia-ml.so / nvml.dll)")
    endif()

    list(APPEND PLUGINS "nvinfer")
    list(APPEND PLUGINS "nvonnxparser")
    list(APPEND PLUGINS "nvparsers")
    list(APPEND PLUGINS "nvinfer_plugin")

    foreach(libName ${PLUGINS})
        find_library(${libName}_lib NAMES ${libName} "/usr" PATH_SUFFIXES lib)
        list(APPEND PLUGIN_LIBS "${${libName}_lib}")
    endforeach()

    FetchContent_Declare(
    gemmi
    GIT_REPOSITORY https://github.com/project-gemmi/gemmi.git
    GIT_TAG        ...
    )
    FetchContent_GetProperties(gemmi)
    if (NOT gemmi_POPULATED)
    FetchContent_Populate(gemmi)
    add_subdirectory(${gemmi_SOURCE_DIR} ${gemmi_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
    FetchContent_MakeAvailable(gemmi)

    # Include glm
    FetchContent_Declare(
            glm
            GIT_REPOSITORY https://github.com/g-truc/glm.git
    )
    FetchContent_MakeAvailable(glm)

    # Include glad
    FetchContent_Declare(
            glad
            GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    )
    FetchContent_MakeAvailable(glad)

    # Include lygia
    FetchContent_Declare(
            lygia
            GIT_REPOSITORY https://github.com/patriciogonzalezvivo/lygia.git
            GIT_TAG "origin/main"
    )
    FetchContent_MakeAvailable(lygia)

    # The executable code is here
    add_subdirectory(apps)

endif()