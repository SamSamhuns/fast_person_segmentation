#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "common.h"
// tensorflowlite headers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/tools/gen_op_registration.h"


int img_mode(Settings settings, char *img_name);
int cam_mode(Settings settings);

int main(int argc, char *argv[]) {
  // run program like
  // for image:  ./example img m(1/2/3/4) path_to_img
  // for webcam: ./example cam m(1/2/3/4)
  // 1st arg is mode, 2nd arg is model
  if (argc == 4 && strcmp(argv[1], "img") == 0) {
    // load model configuration files
    Settings settings = get_settings(argv[2]);
    img_mode(settings, argv[3]);
    return 0;
  } else if (argc == 3 && strcmp(argv[1], "cam") == 0) {
    // load model configuration files
    Settings settings = get_settings(argv[2]);
    cam_mode(settings);
    return 0;
  }
  std::cout << "Invalid number of args. FMT is ./example img m1 <img_path> or "
               "./example cam m1"
            << std::endl;
  return 1;
}

int img_mode(Settings settings, char *img_name) {

  std::cout << "Loading model " << settings.model_path
            << " for img inference on " << img_name << std::endl;
  // Load model settings i.e. path, input dim, & in/out layer names
  const char *model_filepath = settings.model_path.c_str();

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model;
  model = tflite::FlatBufferModel::BuildFromFile(model_filepath);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);
  // set precision for interpreter
  interpreter->SetAllowFp16PrecisionForFp32(settings.allow_fp16);

  if (settings.verbose) {
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

  if (settings.number_of_threads != -1) {
    interpreter->SetNumThreads(settings.number_of_threads);
  }

  // get input/output dimension from the input/output tensor metadata
  int input_index = 0;
  int output_index = 0;
  int input = interpreter->inputs()[input_index];
  TfLiteIntArray *in_dims = interpreter->tensor(input)->dims;
  int wanted_batch_size = in_dims->data[0];
  int wanted_height = in_dims->data[1];
  int wanted_width = in_dims->data[2];
  int wanted_channels = in_dims->data[3];
  int output = interpreter->outputs()[output_index];
  TfLiteIntArray *out_dims = interpreter->tensor(output)->dims;
  int out_batch_size = out_dims->data[0];
  int out_height = out_dims->data[1];
  int out_width = out_dims->data[2];
  int out_channels = out_dims->data[3];

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  if (settings.verbose) {
    printf("##### Printing INPUT/OUTPUT NAME and SHAPES #####\n");
    printf("input shape: [%d,%d,%d,%d]\n", wanted_batch_size, wanted_height, wanted_width, wanted_channels);
    std::cout << "input(0) name: " << interpreter->GetInputName(0) << std::endl;
    std::cout << "number of inputs: " << inputs.size() << std::endl;
    printf("ouput shape: [%d,%d,%d,%d]\n", out_batch_size, out_height, out_width, out_channels);
    std::cout << "output(0) name: " << interpreter->GetOutputName(0) << std::endl;
    std::cout << "number of outputs: " << outputs.size() << std::endl;
  }

  // load iamge with opencv and populate tflite input
  // display window name
  std::string disp_window_name = get_basename(settings.model_path);

  // declare mat vars & input vector
  cv::Mat img, inp, convMat, bgcrop;
  TfLiteTensor* output_mask = nullptr;
  img = cv::imread(img_name, cv::IMREAD_COLOR);
  cv::resize(img, inp, cv::Size(wanted_width, wanted_height), cv::INTER_LINEAR);

  // Scale image to range 0-1 and convert dtype to float 32
  inp.convertTo(inp, CV_32FC3, 1.f / 255);

  // Allocate tensor buffers
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  // printf("=== Pre-invoke Interpreter State ===\n");
  // if (settings.verbose)
  //   tflite::PrintInterpreterState(interpreter.get());

  // https://stackoverflow.com/questions/59424842
  // for input and output buffers:
  //   typed_input_tensor/typed_output_tensor takes the input/output int index
  //   typed_tensor takes interpreter->inputs()[input_idx] or interpreter->outputs()[output_idx]
  // Fill input buffers
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`
  // i.e. float *input = interpreter->typed_input_tensor<float>(0);

  // copy image to input as input tensor
  memcpy(interpreter->typed_input_tensor<float>(input_index), inp.data, inp.total() * inp.elemSize());

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  printf("Tflite Inference Complete\n");
  // printf("=== Post-invoke Interpreter State ===\n");
  // if (settings.verbose)
  //   tflite::PrintInterpreterState(interpreter.get());

  // Read output buffers
  output_mask = interpreter->tensor(interpreter->outputs()[output_index]);
	float* output_data_ptr = output_mask->data.f;
  // TODO
  // int out_data_length = wanted_height * wanted_width * 2;
  // std::vector<float> output_vec {output_data_ptr, output_data_ptr + out_data_length};
  // printf("%lu\n", output_vec.size());
  // get model ouput as vector<float> & conv to cv::Mat & resize to 1,128,128
  // convMat = cv::Mat(output.get_data<float>())
  //               .reshape(1, wanted_height); // channel, num_rows
  //convert from float* to cv::MAT
  cv::Mat two_channel(out_height, out_width, CV_32FC2, output_data_ptr);

  // 0=bg channel, 1=fg channel
  cv::extractChannel(two_channel, convMat, 1);

  // post processing
  cv::cvtColor(convMat, convMat, cv::COLOR_GRAY2BGR);
  cv::GaussianBlur(convMat, convMat, cv::Size(3, 3), 3);
  cv::resize(convMat, convMat, cv::Size(1200, 720), cv::INTER_LINEAR);
  cv::resize(img, img, cv::Size(1200, 720), cv::INTER_LINEAR);
  img.convertTo(img, CV_32FC3);
  convMat.setTo(1.0, convMat >= 0.63);
  convMat.setTo(0.0, convMat < 0.63);
  cv::multiply(convMat, img, convMat); // mix orig img and generated mask
  convMat.convertTo(convMat, CV_8UC3);

  cv::imshow(disp_window_name, convMat);
  cv::waitKey(0);

  return 0;
}

int cam_mode(Settings settings) {
  std::cout << "Loading model " << settings.model_path
            << " for webcam inference " << std::endl;
  // Load model settings i.e. path, input dim, & in/out layer names
  const char *model_filepath = settings.model_path.c_str();

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model;
  model = tflite::FlatBufferModel::BuildFromFile(model_filepath);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  interpreter->SetAllowFp16PrecisionForFp32(settings.allow_fp16);

  if (settings.verbose) {
    std::cout << "tensors size: " << interpreter->tensors_size();
    std::cout << "nodes size: " << interpreter->nodes_size();
    std::cout << "inputs: " << interpreter->inputs().size();
    std::cout << "input(0) name: " << interpreter->GetInputName(0);

    int t_size = interpreter->tensors_size();
    for (int i = 0; i < t_size; i++) {
      if (interpreter->tensor(i)->name)
        std::cout << i << ": " << interpreter->tensor(i)->name << ", "
                  << interpreter->tensor(i)->bytes << ", "
                  << interpreter->tensor(i)->type << ", "
                  << interpreter->tensor(i)->params.scale << ", "
                  << interpreter->tensor(i)->params.zero_point;
    }
  }

  if (settings.number_of_threads != -1) {
    interpreter->SetNumThreads(settings.number_of_threads);
  }

  int input = interpreter->inputs()[0];
  if (settings.verbose)
    std::cout << "input: " << input;

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  if (settings.verbose) {
    std::cout << "number of inputs: " << inputs.size();
    std::cout << "number of outputs: " << outputs.size();
  }

  // Allocate tensor buffers
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  if (settings.verbose)
    tflite::PrintInterpreterState(interpreter.get());

  // get input dimension from the input tensor metadata
  // assuming one input only
  TfLiteIntArray *dims = interpreter->tensor(input)->dims;
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];

  // TODO check opencv video read inference

  // Create a window for display
  std::string disp_window_name = get_basename(settings.model_path);
  namedWindow(disp_window_name, cv::WINDOW_NORMAL);

  // Capture frames from camera
  cv::VideoCapture cap;
  cap.open(0); // Camera

  // declare mat vars & input vector
  cv::Mat frame, orig_frame, convMat;
  std::vector<float> img_data;

  // Process video frames
  while (cv::waitKey(1) < 0) {
    auto start = std::chrono::steady_clock::now();
    // Read frames from camera
    cap >> frame;
    if (frame.empty()) {
      cv::waitKey();
      break;
    }
    orig_frame = frame.clone();
    cv::resize(frame, frame, cv::Size(wanted_width, wanted_height),
               cv::INTER_AREA);

    // Scale image to range 0-1 and convert dtype to float 32
    frame.convertTo(frame, CV_32FC3, 1.f / 255);

    // Put image mat in input vector to be sent into input Tensor
    img_data.assign((float *)frame.data,
                    (float *)frame.data + frame.total() * frame.channels());
    // TODO set input data
    // input.set_data(img_data, {1, wanted_width, wanted_height, 3});

    // run model, output is a Tensor with shape 1,128,128,1

    // get model ouput as vector<float> & conv to cv::Mat & resize to 1,128,128
    // TODO run model inference
    // convMat = cv::Mat(output.get_data<float>())
    //               .reshape(1, wanted_height); // channel, num_rows

    // post processing
    cv::cvtColor(convMat, convMat, cv::COLOR_GRAY2BGR);
    cv::resize(convMat, convMat, cv::Size(1000, 720), cv::INTER_LINEAR);
    cv::resize(orig_frame, orig_frame, cv::Size(1000, 720), cv::INTER_LINEAR);

    orig_frame.convertTo(orig_frame, CV_32FC3);
    cv::GaussianBlur(convMat, convMat, cv::Size(3, 3), 3);
    convMat.setTo(1.0, convMat >= 0.63);
    convMat.setTo(0.0, convMat < 0.63);
    cv::multiply(convMat, orig_frame,
                 convMat); // mix orig img and generated mask
    convMat.convertTo(convMat, CV_8UC3);

    auto end = std::chrono::steady_clock::now();
    // Store the time difference between start and end
    auto diff = std::chrono::duration<double, std::milli>(end - start).count();
    cv::String label =
        cv::format("FPS: %.2f: Inference time %.2f ", 1000.0 / diff, diff);
    cv::putText(convMat, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0));
    cv::imshow(disp_window_name, convMat);
  }
  return 0;
}
