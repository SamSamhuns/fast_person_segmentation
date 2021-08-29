#include "common.h"
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

std::string get_basename(std::string full_path) {
  // get model name from full model path
  std::istringstream stream(full_path);
  std::string str;
  while (std::getline(stream, str, '/')) {
  }
  return str;
}

bool does_file_exist(const std::string &fpath) {
  // if check if a file exists in fpath
  struct stat buffer;
  return (stat(fpath.c_str(), &buffer) == 0);
}

Settings get_settings(std::string model_path) {
  Settings s;
  if (does_file_exist(model_path)) {
    std::cout << "tflite model found at: " << model_path << '\n';
  } else {
    std::cout << "Invalid tflite model path: " << model_path << '\n';
    exit(1);
  }

  s.model_path = model_path;
  return s;
}

void print_model_struct(std::unique_ptr<tflite::Interpreter>& interpreter) {
  std::cout << "Printing model layer name and shapes:" << std::endl;
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
    printf("##### Printing INPUT/OUTPUT NAME and SHAPES #####\n");
    printf("Input shape: [%d,%d,%d,%d]\n", in_bsize, in_height, in_width,
           in_channels);
    std::cout << "input(0) name: " << interpreter->GetInputName(0) << "\n";
    std::cout << "number of inputs: " << inputs.size() << "\n";
    printf("Output shape: [%d,%d,%d,%d]\n", out_bsize, out_height, out_width,
           out_channels);
    std::cout << "output(0) name: " << interpreter->GetOutputName(0) << "\n";
    std::cout << "number of outputs: " << outputs.size() << "\n";
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
