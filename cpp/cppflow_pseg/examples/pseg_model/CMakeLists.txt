cmake_minimum_required(VERSION 3.10)

project(example)

message(STATUS "env home directory: $ENV{HOME}")

set(CMAKE_CXX_STANDARD 17)

find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow/lib)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME}
  main.cpp
  ../../src/Model.cpp
  ../../src/Tensor.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE
  ../../include
  $ENV{HOME}/libtensorflow/include)

target_link_libraries(${PROJECT_NAME}
  "${TENSORFLOW_LIB}"
  ${OpenCV_LIBS}
  )
