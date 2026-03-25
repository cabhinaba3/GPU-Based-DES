cmake_minimum_required(VERSION 3.25)
project(cuda_des LANGUAGES CXX CUDA)

# ─────────────────────────────────────────────────────────────────────────────
#  Language standards
# ─────────────────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ─────────────────────────────────────────────────────────────────────────────
#  CUDA architecture
#  Set to your GPU's compute capability; defaults cover Ampere + Hopper (sm_86+)
#  Override via: cmake -DCMAKE_CUDA_ARCHITECTURES="75;86;89;90"
# ─────────────────────────────────────────────────────────────────────────────
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;86;89;90")
endif()

# ─────────────────────────────────────────────────────────────────────────────
#  Compiler flags
# ─────────────────────────────────────────────────────────────────────────────
set(DES_CUDA_FLAGS
    --extended-lambda          # required for lambda in device code
    --expt-relaxed-constexpr   # constexpr on device
    -lineinfo                  # keep line info for profiling (Nsight)
    -O3
)

# ─────────────────────────────────────────────────────────────────────────────
#  Find packages
# ─────────────────────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ─────────────────────────────────────────────────────────────────────────────
#  Library: des_engine
# ─────────────────────────────────────────────────────────────────────────────
add_library(des_engine STATIC
    src/event_queue.cu
    src/engine.cu
)

target_include_directories(des_engine
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(des_engine
    PUBLIC
        CUDA::cudart
        CUDA::curand
)

target_compile_options(des_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${DES_CUDA_FLAGS}>
)

# Expose separable compilation so callers can link device code
set_target_properties(des_engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON
)

# ─────────────────────────────────────────────────────────────────────────────
#  Library: des_engine_multi  (adds multi-GPU engine on top of des_engine)
# ─────────────────────────────────────────────────────────────────────────────
add_library(des_engine_multi STATIC
    src/multigpu_engine.cu
)

target_include_directories(des_engine_multi
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(des_engine_multi
    PUBLIC
        des_engine       # inherits CUDA::cudart, CUDA::curand, include paths
)

target_compile_options(des_engine_multi PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${DES_CUDA_FLAGS}>
)

set_target_properties(des_engine_multi PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON
)

# ─────────────────────────────────────────────────────────────────────────────
#  Examples
# ─────────────────────────────────────────────────────────────────────────────
# option(BUILD_EXAMPLES "Build example programs" ON)
# if(BUILD_EXAMPLES)
  #  add_subdirectory(examples/mm1)
  # add_subdirectory(examples/sir)
# endif()

# ─────────────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────────────
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
