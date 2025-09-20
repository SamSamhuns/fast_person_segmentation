#ifndef COMMON_H
#define COMMON_H

#include "tensorflow/lite/interpreter.h"
#include <stdio.h>
#include <string>

#define CHECK_FOR_ERROR(x)                                   \
  if (!(x))                                                  \
  {                                                          \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

struct Settings
{
  // init return char ptr vars to nullptr for initialization check later
  char *mode = nullptr;
  char *model_path = nullptr;
  char *in_media_path = nullptr;
  char *bg_path = nullptr;
  char *save_path = nullptr;
  float threshold = 0.8;
  int disp_w = 1200;
  int disp_h = 720;
  bool use_prev_msk = false;
  bool verbose = false;
  // bool accel = false;
  // bool gl_backend = false;
  // bool hexagon_delegate = false;
  // bool xnnpack_delegate = false;
  bool allow_fp16 = false;
  int number_of_threads = 4;
};

struct IOShape
{
  int index;
  int bsize;
  int height;
  int width;
  int channels;
};

Settings get_settings_from_args(int argc, char **argv);
std::string get_basename(std::string full_path);
void print_model_struct(std::unique_ptr<tflite::Interpreter> &interpreter);
std::tuple<IOShape, IOShape>
get_input_output_dims(Settings &settings,
                      std::unique_ptr<tflite::Interpreter> &interpreter);

#endif
