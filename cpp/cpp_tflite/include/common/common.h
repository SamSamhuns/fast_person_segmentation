#ifndef COMMON_H
#define COMMON_H

#include <string>
#include <stdio.h>
#include "tensorflow/lite/interpreter.h"

#define TFLITE_MINIMAL_CHECK(x)                                                \
  if (!(x)) {                                                                  \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);                   \
    exit(1);                                                                   \
  }

struct Settings {
  std::string model_path = "";
  float threshold = 0.63;
  bool verbose = true;
  bool accel = false;
  bool allow_fp16 = false;
  bool gl_backend = false;
  bool hexagon_delegate = false;
  bool xnnpack_delegate = false;
  int number_of_threads = 4;
};

struct IOShape {
  int index;
  int bsize;
  int height;
  int width;
  int channels;
};

Settings get_settings(std::string model_path);
std::string get_basename(std::string full_path);
void print_model_struct(std::unique_ptr<tflite::Interpreter>& interpreter);
std::tuple<IOShape, IOShape>get_input_output_dims(Settings &settings, std::unique_ptr<tflite::Interpreter> &interpreter);

#endif
