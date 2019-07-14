cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(linear)

find_package(Torch REQUIRED)

add_executable(linear linear.cpp)
target_link_libraries(linear "${TORCH_LIBRARIES}")
set_property(TARGET linear PROPERTY CXX_STANDARD 11)
