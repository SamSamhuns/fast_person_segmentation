#ifndef COMMON_H
#define COMMON_H

#include <string>
#include <stdio.h>
#include "tensorflow/lite/interpreter.h"

#define CHECK_FOR_ERROR(x)                                                \
  if (!(x)) {                                                             \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);              \
    exit(1);                                                              \
  }

struct Settings {
  std::string model_path = "";
  char* in_media_path = NULL;
  char* bg_path = NULL;
  char* save_path = NULL;
  float threshold = 0.8;
  int disp_w = 1200;
  int disp_h = 720;
  bool verbose = false;
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

std::tuple<char *, char *, char *, char *, char *, bool> parse_args(int argc, char **argv);
Settings get_settings(std::string model_path, char *in_media_path, char *bg_path, char *save_path, bool verbose);
std::string get_basename(std::string full_path);
void print_model_struct(std::unique_ptr<tflite::Interpreter> &interpreter);
std::tuple<IOShape, IOShape>get_input_output_dims(Settings &settings, std::unique_ptr<tflite::Interpreter> &interpreter);

#endif
