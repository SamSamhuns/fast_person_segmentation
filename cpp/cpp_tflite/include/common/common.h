#ifndef COMMON_H
#define COMMON_H

#include <string>

#define TFLITE_MINIMAL_CHECK(x)                                                \
  if (!(x)) {                                                                  \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);                   \
    exit(1);                                                                   \
  }

struct Settings {
  std::string model_path = "";
  bool verbose = true;
  bool accel = false;
  bool allow_fp16 = false;
  bool gl_backend = false;
  bool hexagon_delegate = false;
  bool xnnpack_delegate = false;
  int number_of_threads = 4;
};

std::string get_basename(std::string full_path);
Settings get_settings(std::string model_type);

#endif
