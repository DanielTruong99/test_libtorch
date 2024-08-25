cmake_minimum_required(VERSION 3.0)
project(custom_ops)

set(CMAKE_PREFIX_PATH "/home/ryz2/DanielWorkspace/libtorch")

find_package(Torch REQUIRED)

add_executable(test_torch main.cpp)
target_link_libraries(test_torch "${TORCH_LIBRARIES}")
