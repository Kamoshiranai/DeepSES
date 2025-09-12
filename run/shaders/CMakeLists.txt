cmake_minimum_required(VERSION 3.14...3.18) #test for versions below 3.18

project(
  uncertainvis
  LANGUAGES CXX C)

include(FetchContent)

set(${SHADER_DIR} "${CMAKE_SOURCE_DIR}/shaders/")


# So far you need to install openbabel yourself and include the path to the directory here as well as further down in LIBS
include_directories(/usr/include/openbabel3)

# Include glfw
FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw
)
FetchContent_MakeAvailable(glfw)

#Include glm
FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
)
FetchContent_MakeAvailable(glm)

#Include glad
FetchContent_Declare(
        glad
        GIT_REPOSITORY https://github.com/Dav1dde/glad.git
)
FetchContent_MakeAvailable(glad)

#Include assimp
FetchContent_Declare(
        assimp
        GIT_REPOSITORY https://github.com/assimp/assimp.git
)
FetchContent_MakeAvailable(assimp)

#enter path to libopenbabel.so here
set(openbabel /usr/lib/x86_64-linux-gnu/libopenbabel.so)

# The executable code is here
add_subdirectory(apps)
add_subdirectory(extern/imgui-master)
