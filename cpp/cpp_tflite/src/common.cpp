#include "common.h"
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
         "--verbose/--v:                 Verbose mode if flag specified\n"
         "--help/-h:                     Show help\n";
}

std::tuple<char *, char *, char *, char *, char *, bool> parse_args(int argc,
                                                              char **argv) {
  // init return char ptr vars to nullptr for initialization check later
  char *mode = nullptr;
  char *tflite_model_path = nullptr;
  char *in_media_path = nullptr;
  char *bg_image_path = nullptr;
  char *save_path = nullptr;
  bool verbose = false;

  const char *const short_opts = "m:t:i:b:s:h";
  const option long_opts[] = {
      {"mode", required_argument, nullptr, 'm'},
      {"tflite_model_path", required_argument, nullptr, 't'},
      {"in_media_path", required_argument, nullptr, 'i'},
      {"bg_image_path", required_argument, nullptr, 'b'},
      {"save_path", required_argument, nullptr, 's'},
      {"verbose", no_argument, nullptr, 'v'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, no_argument, nullptr, 0}};

  while (true) {
    const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);
    if (-1 == opt)
      break;

    switch (opt) {
    case 'm':
      mode = optarg;
      std::cout << "Inference mode: " << mode << "\n";
      break;
    case 't':
      tflite_model_path = optarg;
      std::cout << "TFlite model path: " << tflite_model_path << "\n";
      break;
    case 'i':
      in_media_path = optarg;
      std::cout << "Input media path: " << in_media_path << "\n";
      break;
    case 'b':
      bg_image_path = optarg;
      std::cout << "Background image path: " << bg_image_path << "\n";
      break;
    case 's':
      save_path = optarg;
      std::cout << "Media save path: " << save_path << "\n";
      break;
    case 'v':
      verbose = true;
      std::cout << "Verbose mode: " << verbose << "\n";
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
  if ((mode == nullptr) || (mode[0] == '\0')) {
    std::cout << "ERROR: -m img/vid must be specified. Use -h for help" << '\n';
    exit(-1);
  } else if ((tflite_model_path == nullptr) || (tflite_model_path[0] == '\0')) {
    std::cout << "ERROR: tflite model path is needed. Use -h for help \n";
    exit(-1);
  }
  return std::make_tuple(mode, tflite_model_path, in_media_path, bg_image_path,
                         save_path, verbose);
}

std::string get_basename(std::string full_path) {
  // get model name from full model path
  std::istringstream stream(full_path);
  std::string str;
  while (std::getline(stream, str, '/')) {
  }
  return str;
}

bool does_file_exist(const char *fpath) {
  // if check if a file exists in fpath
  struct stat buffer;
  return (stat(fpath, &buffer) == 0);
}

Settings get_settings(std::string model_path, char *in_media_path,
                      char *bg_path, char *save_path, bool verbose) {
  Settings s;
  // if the paths are not NULL and are valid
  if (!does_file_exist(model_path.c_str())) {
    std::cout << "ERROR: Invalid tflite model path: " << model_path << '\n';
    exit(1);
  } else if (in_media_path != nullptr && !does_file_exist(in_media_path)) {
    std::cout << "ERROR: Invalid in media path: " << in_media_path << '\n';
    exit(1);
  } else if (bg_path != nullptr && !does_file_exist(bg_path)) {
    std::cout << "ERROR: Invalid bg image path: " << bg_path << '\n';
    exit(1);
  }

  s.in_media_path = in_media_path;
  s.bg_path = bg_path;
  s.save_path = save_path;
  s.model_path = model_path;
  s.verbose = verbose;
  return s;
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
