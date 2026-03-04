
if(WIN32)
    # ----------------------------------------
    # WINDOWS
    # ----------------------------------------

    add_executable(create_engine create_engine.cc)
    # target_compile_features(create_engine PRIVATE cxx_std_11)
    target_compile_features(create_engine PRIVATE cxx_std_17)

    target_link_libraries(create_engine PRIVATE ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${TensorRT_NVONNXPARSER_LIBRARY} ${HDF5_LIBRARIES})
    target_include_directories(create_engine PRIVATE ../include ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIR})

    add_executable(deepses_with_noise deepses_with_noise.cc thrust_wrapper.cu filter_patches_kernels.cu ../shaders/raymarch_get_visited_patches.cu)
    # target_compile_features(deepses_with_noise PRIVATE cxx_std_11)
    target_compile_features(deepses_with_noise PRIVATE cxx_std_17)

    target_compile_options(deepses_with_noise PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>) # for printf

    target_link_libraries(deepses_with_noise PRIVATE glfw glad glm gemmi::gemmi_cpp ZLIB::ZLIB ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${HDF5_LIBRARIES}) #  ${nvinfer_lib}
    set_target_properties(deepses_with_noise PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # for printf
    target_include_directories(deepses_with_noise PRIVATE ../include ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIR})

    # Copy all shaders into the build dir everytime the code is compiled.
    add_custom_target(copy_shaders ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/shaders" "${CMAKE_CURRENT_BINARY_DIR}/shaders"
        COMMENT "Copy shaders to build tree"
        VERBATIM)

    add_dependencies(deepses_with_noise copy_shaders)

else()

    # ----------------------------------------
    # LINUX
    # ----------------------------------------

    add_executable(create_engine create_engine.cc)
    target_compile_features(create_engine PRIVATE cxx_std_11)

    target_link_libraries(create_engine PRIVATE ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${nvinfer_lib} ${nvonnxparser_lib}) #CUDA::cudart not working?
    target_include_directories(create_engine PRIVATE ../include ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})

    add_executable(deepses_with_noise deepses_with_noise.cc thrust_wrapper.cu filter_patches_kernels.cu ../shaders/raymarch_get_visited_patches.cu)
    target_compile_features(deepses_with_noise PRIVATE cxx_std_11)

    target_compile_options(deepses_with_noise PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>) # for printf

    target_link_libraries(deepses_with_noise PRIVATE glfw glad glm gemmi::gemmi_cpp ZLIB::ZLIB ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${nvinfer_lib})
    set_target_properties(deepses_with_noise PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # for printf
    target_include_directories(deepses_with_noise PRIVATE ../include ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})

    # Copy all shaders into the build dir everytime the code is compiled.
    add_custom_target(copy_shaders ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/shaders" "${CMAKE_CURRENT_BINARY_DIR}/shaders"
        COMMENT "Copy shaders to build tree"
        VERBATIM)

    add_dependencies(deepses_with_noise copy_shaders)

endif()