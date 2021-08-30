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

int img_mode(Settings &settings, char *img_path);
int cam_mode(Settings &settings, char *vid_path);

int main(int argc, char *argv[]) {
  // run program like
  // for image:  ./example img m(1/2/3/4) path_to_img
  // for webcam: ./example cam m(1/2/3/4)
  // 1st arg is mode, 2nd arg is model_path
  if (argc == 4 && strcmp(argv[1], "img") == 0) {
    Settings settings = get_settings(argv[2]);
    img_mode(settings, argv[3]);
    return 0;
  } else if (argc == 4 && strcmp(argv[1], "cam") == 0) {
    Settings settings = get_settings(argv[2]);
    cam_mode(settings, argv[3]);
    return 0;
  }
  std::cout << "Invalid number of args. FMT is ./example img "
               "<tflite_model_path> <img_path> or "
               "./example cam <tflite_model_path>"
            << std::endl;
  return 1;
}

int img_mode(Settings &settings, char *img_path) {
  std::cout << "Starting image inference on " << img_path << "\n";
  const char *model_filepath = settings.model_path.c_str();

  // load model and get tflite interpreter
  std::unique_ptr<tflite::FlatBufferModel> model;
  model = tflite::FlatBufferModel::BuildFromFile(model_filepath);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  interpreter->SetAllowFp16PrecisionForFp32(settings.allow_fp16);
  if (settings.number_of_threads != -1) {
    interpreter->SetNumThreads(settings.number_of_threads);
  }
  print_model_struct(interpreter);

  // get input/output shapes
  IOShape input_shape;
  IOShape output_shape;
  std::tie(input_shape, output_shape) =
      get_input_output_dims(settings, interpreter);

  // load iamge with opencv and populate tflite input
  // display window name
  std::string disp_window_name = get_basename(settings.model_path);

  // declare mat vars & input vector
  cv::Mat img, inp, convMat, bgcrop;
  TfLiteTensor *output_mask = nullptr;
  img = cv::imread(img_path, cv::IMREAD_COLOR);
  cv::resize(img, inp, cv::Size(input_shape.width, input_shape.height),
             cv::INTER_LINEAR);
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
  //   typed_tensor takes interpreter->inputs()[input_idx] or
  //   interpreter->outputs()[output_idx]

  // copy image to input as input tensor
  memcpy(interpreter->typed_input_tensor<float>(input_shape.index), inp.data,
         inp.total() * inp.elemSize());

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  printf("Tflite Inference Complete\n");
  // printf("=== Post-invoke Interpreter State ===\n");
  // if (settings.verbose)
  //   tflite::PrintInterpreterState(interpreter.get());

  // Read output buffers
  output_mask = interpreter->tensor(interpreter->outputs()[output_shape.index]);
  float *output_data_ptr = output_mask->data.f;

  // convert from float* to cv::mat
  cv::Mat two_channel(output_shape.height, output_shape.width, CV_32FC2,
                      output_data_ptr);

  // 0=bg channel, 1=fg channel
  cv::extractChannel(two_channel, convMat, 1);

  // post processing
  cv::cvtColor(convMat, convMat, cv::COLOR_GRAY2BGR);
  cv::GaussianBlur(convMat, convMat, cv::Size(3, 3), 3);
  cv::resize(convMat, convMat, cv::Size(1200, 720), cv::INTER_LINEAR);
  cv::resize(img, img, cv::Size(1200, 720), cv::INTER_LINEAR);
  img.convertTo(img, CV_32FC3);
  convMat.setTo(1.0, convMat >= settings.threshold);
  convMat.setTo(0.0, convMat < settings.threshold);
  cv::multiply(convMat, img, convMat); // mix orig img and generated mask
  convMat.convertTo(convMat, CV_8UC3);

  cv::imshow(disp_window_name, convMat);
  cv::waitKey(0);

  return 0;
}

int cam_mode(Settings &settings, char *vid_path) {
  std::cout << "Initializing video inference\n";
  const char *model_filepath = settings.model_path.c_str();

  // load model and get tflite interpreter
  std::unique_ptr<tflite::FlatBufferModel> model;
  model = tflite::FlatBufferModel::BuildFromFile(model_filepath);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  interpreter->SetAllowFp16PrecisionForFp32(settings.allow_fp16);
  if (settings.number_of_threads != -1) {
    interpreter->SetNumThreads(settings.number_of_threads);
  }
  print_model_struct(interpreter);

  // get input/output shapes
  IOShape input_shape;
  IOShape output_shape;
  std::tie(input_shape, output_shape) =
      get_input_output_dims(settings, interpreter);

  // Create a window for display
  std::string disp_window_name = get_basename(settings.model_path);
  namedWindow(disp_window_name, cv::WINDOW_NORMAL);

  // Capture frames from video
  cv::VideoCapture cap;
  if (vid_path[0] == '0') { // webcam mode
    cap.open(0);
  }
  else {
    cap.open(vid_path);     // video loaded from path
  }

  // declare mat vars & input vector
  cv::Mat frame, orig_frame, convMat;
  TfLiteTensor *output_mask = nullptr;
  // Allocate tensor buffers
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

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
    cv::resize(frame, frame, cv::Size(input_shape.width, input_shape.height),
               cv::INTER_AREA);

    // Scale image to range 0-1 and convert dtype to float 32
    frame.convertTo(frame, CV_32FC3, 1.f / 255);

    // copy image to input as input tensor
    memcpy(interpreter->typed_input_tensor<float>(input_shape.index), frame.data,
           frame.total() * frame.elemSize());

    // Run inference
    interpreter->Invoke();

    // Read output buffers
    output_mask = interpreter->tensor(interpreter->outputs()[output_shape.index]);
    float *output_data_ptr = output_mask->data.f;

    // convert from float* to cv::mat
    cv::Mat two_channel(output_shape.height, output_shape.width, CV_32FC2,
                        output_data_ptr);

    // 0=bg channel, 1=fg channel
    cv::extractChannel(two_channel, convMat, 1);

    // post processing
    cv::cvtColor(convMat, convMat, cv::COLOR_GRAY2BGR);
    cv::resize(convMat, convMat, cv::Size(1000, 720), cv::INTER_LINEAR);
    cv::resize(orig_frame, orig_frame, cv::Size(1000, 720), cv::INTER_LINEAR);

    orig_frame.convertTo(orig_frame, CV_32FC3);
    cv::GaussianBlur(convMat, convMat, cv::Size(3, 3), 3);
    convMat.setTo(1.0, convMat >= settings.threshold);
    convMat.setTo(0.0, convMat < settings.threshold);
    cv::multiply(convMat, orig_frame, convMat); // mix orig img and mask
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
