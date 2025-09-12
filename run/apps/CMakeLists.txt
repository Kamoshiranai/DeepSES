
if(WIN32)
    # ----------------------------------------
    # WINDOWS
    # ----------------------------------------

    add_executable(interactive-hermosilla-method interactive-hermosilla-method.cc)
    # target_compile_features(interactive-hermosilla-method PRIVATE cxx_std_11)
    target_compile_features(interactive-hermosilla-method PRIVATE cxx_std_17)

    target_link_libraries(interactive-hermosilla-method PRIVATE glfw glad glm gemmi::gemmi_cpp ZLIB::ZLIB ${NVML_LIBRARY})
    target_include_directories(interactive-hermosilla-method PRIVATE ../include  ${CUDA_INCLUDE_DIRS})

    add_executable(create_engine create_engine.cc)
    # target_compile_features(create_engine PRIVATE cxx_std_11)
    target_compile_features(create_engine PRIVATE cxx_std_17)

    target_link_libraries(create_engine PRIVATE ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${TensorRT_NVONNXPARSER_LIBRARY} ${HDF5_LIBRARIES})
    target_include_directories(create_engine PRIVATE ../include ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIR})

    add_executable(interactive-deepses interactive-deepses.cc thrust_wrapper.cu reshape_kernels.cu filter_patches_kernels.cu ../shaders/raymarch_get_visited_patches.cu)
    # target_compile_features(interactive-deepses PRIVATE cxx_std_11)
    target_compile_features(interactive-deepses PRIVATE cxx_std_17)

    target_compile_options(interactive-deepses PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>) # for printf

    target_link_libraries(interactive-deepses PRIVATE glfw glad glm gemmi::gemmi_cpp ZLIB::ZLIB ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${NVML_LIBRARY}) #  ${nvinfer_lib}
    set_target_properties(interactive-deepses PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # for printf
    target_include_directories(interactive-deepses PRIVATE ../include ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIR})

    add_executable(interactive-deepses_with_ao interactive-deepses_with_ao.cc thrust_wrapper.cu reshape_kernels.cu filter_patches_kernels.cu ../shaders/raymarch_get_visited_patches.cu)
    # target_compile_features(interactive-deepses_with_ao PRIVATE cxx_std_11)
    target_compile_features(interactive-deepses_with_ao PRIVATE cxx_std_17)

    target_compile_options(interactive-deepses_with_ao PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>) # for printf

    target_link_libraries(interactive-deepses_with_ao PRIVATE glfw glad glm gemmi::gemmi_cpp ZLIB::ZLIB ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${HDF5_LIBRARIES}) #  ${nvinfer_lib}
    set_target_properties(interactive-deepses_with_ao PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # for printf
    target_include_directories(interactive-deepses_with_ao PRIVATE ../include ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIR})

    add_executable(interactive-vdw interactive-vdw.cc)
    target_compile_features(vdw PRIVATE cxx_std_17)
    # target_compile_features(vdw PRIVATE cxx_std_11)

    target_link_libraries(interactive-vdw PRIVATE glfw glad glm gemmi::gemmi_cpp ZLIB::ZLIB ${HDF5_LIBRARIES})
    target_include_directories(interactive-vdw PRIVATE ../include)

    # Copy all shaders into the build dir everytime the code is compiled.
    add_custom_target(copy_shaders ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/shaders" "${CMAKE_CURRENT_BINARY_DIR}/shaders"
        COMMENT "Copy shaders to build tree"
        VERBATIM)

    add_dependencies(interactive_deepses copy_shaders)
    add_dependencies(interactive_deepses_with_ao copy_shaders)
    add_dependencies(interactive_hermosilla_method copy_shaders)
    add_dependencies(interactive_vdw copy_shaders)

else()

    # ----------------------------------------
    # LINUX
    # ----------------------------------------

    add_executable(interactive-hermosilla-method interactive-hermosilla-method.cc)
    target_compile_features(interactive-hermosilla-method PRIVATE cxx_std_11)

    target_link_libraries(interactive-hermosilla-method PRIVATE glfw glad glm gemmi::gemmi_cpp ZLIB::ZLIB ${NVML_LIBRARY})
    target_include_directories(interactive-hermosilla-method PRIVATE ../include ${CUDA_INCLUDE_DIRS}) # CUDA_INCLUDE_DIRS only necessary for NVML

    add_executable(create_engine create_engine.cc)
    target_compile_features(create_engine PRIVATE cxx_std_11)

    target_link_libraries(create_engine PRIVATE ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${nvinfer_lib} ${nvonnxparser_lib}) #CUDA::cudart not working?
    target_include_directories(create_engine PRIVATE ../include ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})

    add_executable(interactive-deepses interactive-deepses.cc thrust_wrapper.cu reshape_kernels.cu filter_patches_kernels.cu ../shaders/raymarch_get_visited_patches.cu)
    target_compile_features(interactive-deepses PRIVATE cxx_std_11)

    target_compile_options(interactive-deepses PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>) # for printf

    target_link_libraries(interactive-deepses PRIVATE glfw glad glm gemmi::gemmi_cpp ZLIB::ZLIB ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${nvinfer_lib} ${NVML_LIBRARY})
    set_target_properties(interactive-deepses PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # for printf
    target_include_directories(interactive-deepses PRIVATE ../include ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})

    add_executable(interactive-deepses_with_ao interactive-deepses_with_ao.cc thrust_wrapper.cu reshape_kernels.cu filter_patches_kernels.cu ../shaders/raymarch_get_visited_patches.cu)
    target_compile_features(interactive-deepses_with_ao PRIVATE cxx_std_11)

    target_compile_options(interactive-deepses_with_ao PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>) # for printf

    target_link_libraries(interactive-deepses_with_ao PRIVATE glfw glad glm gemmi::gemmi_cpp ZLIB::ZLIB ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${nvinfer_lib})
    set_target_properties(interactive-deepses_with_ao PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # for printf
    target_include_directories(interactive-deepses_with_ao PRIVATE ../include ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})

    add_executable(interactive-vdw interactive-vdw.cc)
    target_compile_features(vdw PRIVATE cxx_std_11)

    target_link_libraries(interactive-vdw PRIVATE glfw glad glm gemmi::gemmi_cpp ZLIB::ZLIB)
    target_include_directories(interactive-vdw PRIVATE ../include)

    # Copy all shaders into the build dir everytime the code is compiled.
    add_custom_target(copy_shaders ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/shaders" "${CMAKE_CURRENT_BINARY_DIR}/shaders"
        COMMENT "Copy shaders to build tree"
        VERBATIM)

    add_dependencies(interactive_deepses copy_shaders)
    add_dependencies(interactive_deepses_with_ao copy_shaders)
    add_dependencies(interactive_hermosilla_method copy_shaders)
    add_dependencies(interactive_vdw copy_shaders)

endif()