cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(eye)

find_package(Torch REQUIRED)

add_executable(eye eye.cpp)
target_link_libraries(eye "${TORCH_LIBRARIES}")
set_property(TARGET eye PROPERTY CXX_STANDARD 11)
