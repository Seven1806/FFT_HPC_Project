# Specify the minimum version of CMake
cmake_minimum_required(VERSION 3.18)

# Project name
project(FFT_Performance LANGUAGES C CXX CUDA)

# Set build type to Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Enable OpenMP support
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -fopenmp -ffast-math -funroll-loops")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -fopenmp -ffast-math -funroll-loops")

# Enable CUDA support
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES 80)  # NVIDIA A100 GPUs
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -Xcompiler -fopenmp")

# Source directory
set(SRC_DIR "/home/alunos/tei/2024/tei27387/hpc_project/src")

# Define source files
set(SEQ_SRC ${SRC_DIR}/fft_sequencial.c)
set(PAR_SRC ${SRC_DIR}/fft_parallel.c)
set(GPU_SRC ${SRC_DIR}/fft_gpu.cu)
set(SPEEDS_SRC ${SRC_DIR}/fft_4speeds_clean.cu)

# Add executables
add_executable(fft_sequencial ${SEQ_SRC})
add_executable(fft_parallel ${PAR_SRC})
add_executable(fft_gpu ${GPU_SRC})
add_executable(fft_4speeds ${SPEEDS_SRC})

# Link OpenMP and Math library to CPU implementations
target_link_libraries(fft_parallel OpenMP::OpenMP_C m)
target_link_libraries(fft_sequencial OpenMP::OpenMP_C m)
target_link_libraries(fft_4speeds OpenMP::OpenMP_C m)

# Link CUDA libraries to the GPU implementation
set_target_properties(fft_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(fft_gpu cufft cudart)

# Ensure correct CUDA policies
cmake_policy(SET CMP0104 NEW)

# Print configuration summary
message(STATUS "CMake Configuration Complete! Ready to Build.")
