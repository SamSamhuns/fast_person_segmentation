cmake_minimum_required(VERSION 3.20)

project("tflite_seg")

set(CMAKE_CXX_STANDARD 17)
set(CPP_SOURCES
  "src/webcam_tflite.cpp"
  "src/common.cpp")

add_subdirectory(
  "include/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
)
# get opencv lib
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME}
  ${CPP_SOURCES})

# add include dirs with common header
target_include_directories(${PROJECT_NAME} PRIVATE
  "include/common")

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  tensorflow-lite)
