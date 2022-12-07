cmake_minimum_required(VERSION 3.24)
project(wmma CUDA)

set(CMAKE_CUDA_STANDARD 17)

# CUDA
include_directories(/usr/local/cuda/include /usr/local/cuda/samples/common/inc)
link_directories(/usr/local/cuda/lib64/)

# Main
add_executable(wmma main.cu)
target_link_libraries(wmma -lstdc++ -lcudart -lcudnn -lcurand -lcublas)
set_target_properties(wmma PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
