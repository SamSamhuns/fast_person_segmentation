#include "common.hpp"
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <tuple>
#include <unordered_map>
// tf header files
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/tools/gen_op_registration.h"

void print_help() {
  std::cout
      << "--mode/-m <img/vid>:           Use -m img/vid for image/video mode\n"
         "--tflite_model_path/-t <path>: Path to tflite model\n"
         "--in_media_path/-i <path>:     Path to fg img/vid given mode\n"
         "--bg_image_path/-b <path>:     Path to bg image."
         "If absent, use a black background\n"
         "--save_path/-s <path>:         Path to save inference image/video."
         "If absent, no results saved\n"
         "--use_prev_msk/--p:            Use previous mask for inference "
         "stability."
         "Might reduce FPS\n"
         "--verbose/--v:                 Verbose mode if flag specified\n"
         "--help/-h:                     Show help\n";
}

bool does_file_exist(const char *fpath) {
  // if check if a file exists in fpath
  struct stat buffer;
  return (stat(fpath, &buffer) == 0);
}

Settings get_settings_from_args(int argc, char **argv) {
  Settings s;

  const char *const short_opts = "m:t:i:b:s:h";
  const option long_opts[] = {
      {"mode", required_argument, nullptr, 'm'},
      {"model_path", required_argument, nullptr, 't'},
      {"in_media_path", required_argument, nullptr, 'i'},
      {"bg_image_path", required_argument, nullptr, 'b'},
      {"save_path", required_argument, nullptr, 's'},
      {"use_prev_msk", no_argument, nullptr, 'p'},
      {"verbose", no_argument, nullptr, 'v'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, no_argument, nullptr, 0}};

  while (true) {
    const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);
    if (-1 == opt)
      break;

    switch (opt) {
    case 'm':
      s.mode = optarg;
      std::cout << "Inference mode: " << s.mode << "\n";
      break;
    case 't':
      s.model_path = optarg;
      std::cout << "TFlite model path: " << s.model_path << "\n";
      break;
    case 'i':
      s.in_media_path = optarg;
      std::cout << "Input media path: " << s.in_media_path << "\n";
      break;
    case 'b':
      s.bg_path = optarg;
      std::cout << "Background image path: " << s.bg_path << "\n";
      break;
    case 's':
      s.save_path = optarg;
      std::cout << "Media save path: " << s.save_path << "\n";
      break;
    case 'p':
      s.use_prev_msk = true;
      std::cout << "Use previous mask: " << s.use_prev_msk << "\n";
      break;
    case 'v':
      s.verbose = true;
      std::cout << "Verbose mode: " << s.verbose << "\n";
      break;

    case 'h': // -h or --help
    case '?': // Unrecognized option
    default:
      print_help();
      exit(1);
    }
  }

  // mode and tflite_model_path must not be nullptr
  // note for this check to happen, vars must be initialized to nullptr
  if ((s.mode == nullptr) || (s.mode[0] == '\0')) {
    std::cout << "ERROR: -m img/vid must be specified. Use -h for help" << '\n';
    exit(1);
  } else if ((s.model_path == nullptr) || (s.model_path[0] == '\0')) {
    std::cout << "ERROR: tflite model path is needed. Use -h for help \n";
    exit(1);
  }

  // if the paths are not NULL and are valid
  if (!does_file_exist(s.model_path)) {
    std::cout << "ERROR: Invalid tflite model path: " << s.model_path << '\n';
    exit(1);
  } else if (s.in_media_path != nullptr && !does_file_exist(s.in_media_path)) {
    std::cout << "ERROR: Invalid in media path: " << s.in_media_path << '\n';
    exit(1);
  } else if (s.bg_path != nullptr && !does_file_exist(s.bg_path)) {
    std::cout << "ERROR: Invalid bg image path: " << s.bg_path << '\n';
    exit(1);
  }
  return s;
}

std::string get_basename(std::string full_path) {
  // get model name from full model path
  std::istringstream stream(full_path);
  std::string str;
  while (std::getline(stream, str, '/')) {
  }
  return str;
}

void print_model_struct(std::unique_ptr<tflite::Interpreter> &interpreter) {
  std::cout << "INFO: Printing model layer name and shapes:" << std::endl;
  int t_size = interpreter->tensors_size();
  for (int i = 0; i < t_size; i++) {
    if (interpreter->tensor(i)->name)
      std::cout << i << ": " << interpreter->tensor(i)->name << ", "
                << interpreter->tensor(i)->bytes << ", "
                << interpreter->tensor(i)->type << ", "
                << interpreter->tensor(i)->params.scale << ", "
                << interpreter->tensor(i)->params.zero_point << std::endl;
  }
}

std::tuple<IOShape, IOShape>
get_input_output_dims(Settings &settings,
                      std::unique_ptr<tflite::Interpreter> &interpreter) {
  // get input/output dimension from the input/output tensor metadata
  int input_index = 0;
  int output_index = 0;
  int input = interpreter->inputs()[input_index];
  TfLiteIntArray *in_dims = interpreter->tensor(input)->dims;
  int in_bsize = in_dims->data[0];
  int in_height = in_dims->data[1];
  int in_width = in_dims->data[2];
  int in_channels = in_dims->data[3];
  int output = interpreter->outputs()[output_index];
  TfLiteIntArray *out_dims = interpreter->tensor(output)->dims;
  int out_bsize = out_dims->data[0];
  int out_height = out_dims->data[1];
  int out_width = out_dims->data[2];
  int out_channels = out_dims->data[3];

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  if (settings.verbose) {
    printf("INFO: Printing Model input/output names and shapes\n");
    printf("\tInput shape: [%d,%d,%d,%d]\n", in_bsize, in_height, in_width,
           in_channels);
    std::cout << "\tinput(0) name: " << interpreter->GetInputName(0) << "\n";
    std::cout << "\tnumber of inputs: " << inputs.size() << "\n";
    printf("\tOutput shape: [%d,%d,%d,%d]\n", out_bsize, out_height, out_width,
           out_channels);
    std::cout << "\toutput(0) name: " << interpreter->GetOutputName(0) << "\n";
    std::cout << "\tnumber of outputs: " << outputs.size() << "\n";
  }

  IOShape input_shape;
  IOShape output_shape;

  input_shape.index = input_index;
  input_shape.bsize = in_bsize;
  input_shape.height = in_height;
  input_shape.width = in_width;
  input_shape.channels = in_channels;

  output_shape.index = output_index;
  output_shape.bsize = out_bsize;
  output_shape.height = out_height;
  output_shape.width = out_width;
  output_shape.channels = out_channels;
  return std::make_tuple(input_shape, output_shape);
}
