cmake_minimum_required(VERSION 3.30)
project(tma_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

include(FetchContent)
FetchContent_Declare(
  nvbench
  GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
  GIT_TAG main)
FetchContent_MakeAvailable(nvbench)

add_executable(tma_benchmark tma.cu)

target_link_libraries(tma_benchmark PRIVATE nvbench::nvbench)

target_compile_options(
  tma_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
                        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
