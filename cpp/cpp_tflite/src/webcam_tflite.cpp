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

int img_mode(Settings &settings);
int vid_mode(Settings &settings);

int main(int argc, char *argv[]) {
  // parse args and fill vars
  // in_media_path, bg_image_path and save_path can be nullptr pointers
  char *mode, *model_path, *in_media_path, *bg_path, *save_path;
  bool verbose;
  std::tie(mode, model_path, in_media_path, bg_path, save_path, verbose) =
      parse_args(argc, argv);

  Settings settings =
      get_settings(model_path, in_media_path, bg_path, save_path, verbose);

  if (strcmp(mode, "img") == 0) {
    if ((in_media_path == nullptr) || (in_media_path[0] == '\0')) {
      std::cout << "ERROR: Input image path is needed. Use -h for help\n";
      exit(1);
    }
    img_mode(settings);
  } else if (strcmp(mode, "vid") == 0) {
    vid_mode(settings);
  }
  return 0;
}

int img_mode(Settings &settings) {
  const char *model_filepath = settings.model_path.c_str();
  char *in_media_path = settings.in_media_path;
  char *bg_path = settings.bg_path;
  char *save_path = settings.save_path;

  // cap min and max display heights
  int disp_h = std::min(720, settings.disp_h);
  int disp_w = std::min(1200, settings.disp_w);

  // load model and get tflite interpreter
  std::unique_ptr<tflite::FlatBufferModel> model;
  model = tflite::FlatBufferModel::BuildFromFile(model_filepath);
  CHECK_FOR_ERROR(model != nullptr);

  // Build the interpreter with the InterpreterBuilder
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  CHECK_FOR_ERROR(interpreter != nullptr);

  interpreter->SetAllowFp16PrecisionForFp32(settings.allow_fp16);
  if (settings.number_of_threads != -1) {
    interpreter->SetNumThreads(settings.number_of_threads);
  }
  if (settings.verbose)
    print_model_struct(interpreter);

  // get input/output shapes
  IOShape in_shape;
  IOShape out_shape;
  std::tie(in_shape, out_shape) = get_input_output_dims(settings, interpreter);

  // load image with opencv and populate tflite input
  // display window name
  std::string disp_window_name = get_basename(settings.model_path);

  // declare mat vars & input vector
  cv::Mat img, inp, convMat, bgcrop;
  TfLiteTensor *output_mask = nullptr;
  // Allocate tensor buffers
  CHECK_FOR_ERROR(interpreter->AllocateTensors() == kTfLiteOk);

  // for removing border feathers
  int kern_size = 1;
  // MORPH_RECT, MORPH_ELLIPSE
  cv::Mat element = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(2 * kern_size + 1, 2 * kern_size + 1),
      cv::Point(kern_size, kern_size));

  // load bg image for replacement
  cv::Mat bg, bg_mask;
  if ((bg_path == nullptr) ||
      (bg_path[0] == '\0')) { // if no bg img path provided, use a zero image
    bg = cv::Mat::zeros(cv::Size(disp_w, disp_h), CV_32FC3);
  } else {
    bg = cv::imread(bg_path); // if bg img path provided, load as BGR
    cv::resize(bg, bg, cv::Size(disp_w, disp_h), cv::INTER_LINEAR);
    bg.convertTo(bg, CV_32FC3);
  }

  // read input image and preprocess for inference
  img = cv::imread(in_media_path, cv::IMREAD_COLOR);
  cv::resize(img, inp, cv::Size(in_shape.width, in_shape.height),
             cv::INTER_LINEAR);
  // Scale image to range 0-1 and convert dtype to float 32
  inp.convertTo(inp, CV_32FC3, 1.f / 255);

  // https://stackoverflow.com/questions/59424842
  // for input and output buffers:
  //   typed_input_tensor/typed_output_tensor takes the input/output int index
  //   typed_tensor takes interpreter->inputs()[input_idx] or
  //   interpreter->outputs()[output_idx]

  // copy image to input as input tensor
  memcpy(interpreter->typed_input_tensor<float>(in_shape.index), inp.data,
         inp.total() * inp.elemSize());

  // Run inference
  CHECK_FOR_ERROR(interpreter->Invoke() == kTfLiteOk);
  printf("Tflite Inference Complete\n");

  // Read output buffers
  output_mask = interpreter->tensor(interpreter->outputs()[out_shape.index]);
  float *output_data_ptr = output_mask->data.f;

  // convert from float* to cv::mat
  cv::Mat two_channel(out_shape.height, out_shape.width, CV_32FC2,
                      output_data_ptr);

  // 0=bg channel, 1=fg channel
  cv::extractChannel(two_channel, convMat, 0);

  // post processing
  cv::dilate(convMat, convMat, element);
  cv::GaussianBlur(convMat, convMat, cv::Size(3, 3), 3);
  cv::resize(convMat, convMat, cv::Size(disp_w, disp_h), cv::INTER_LINEAR);

  cv::threshold(convMat, convMat, settings.threshold, 1.,
                cv::THRESH_BINARY_INV);
  cv::cvtColor(convMat, convMat, cv::COLOR_GRAY2BGR);
  cv::subtract(bg, convMat * 255.0, bg_mask);

  cv::resize(img, img, cv::Size(disp_w, disp_h), cv::INTER_LINEAR);
  img.convertTo(img, CV_32FC3);

  cv::multiply(convMat, img, convMat); // mix orig img and generated mask

  convMat.convertTo(convMat, CV_8UC3);
  bg_mask.convertTo(bg_mask, CV_8UC3);
  cv::add(convMat, bg_mask, convMat);

  cv::imshow(disp_window_name, convMat);
  // save output save_path if provided
  if ((save_path != nullptr) && (save_path[0] != '\0')) {
    cv::imwrite(save_path, convMat);
  }
  cv::waitKey(0);
  return 0;
}

int vid_mode(Settings &settings) {
  const char *model_filepath = settings.model_path.c_str();
  char *in_media_path = settings.in_media_path;
  char *bg_path = settings.bg_path;
  char *save_path = settings.save_path;

  // cap min and max display heights
  int disp_h = std::min(720, settings.disp_h);
  int disp_w = std::min(1000, settings.disp_w);

  // load model and get tflite interpreter
  std::unique_ptr<tflite::FlatBufferModel> model;
  model = tflite::FlatBufferModel::BuildFromFile(model_filepath);
  CHECK_FOR_ERROR(model != nullptr);

  // Build the interpreter with the InterpreterBuilder
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  CHECK_FOR_ERROR(interpreter != nullptr);

  interpreter->SetAllowFp16PrecisionForFp32(settings.allow_fp16);
  if (settings.number_of_threads != -1) {
    interpreter->SetNumThreads(settings.number_of_threads);
  }
  if (settings.verbose)
    print_model_struct(interpreter);

  // get input/output shapes
  IOShape in_shape;
  IOShape out_shape;
  std::tie(in_shape, out_shape) = get_input_output_dims(settings, interpreter);

  // Create a window for display
  std::string disp_window_name = get_basename(settings.model_path);
  namedWindow(disp_window_name, cv::WINDOW_NORMAL);

  // Capture frames from video
  cv::VideoCapture cap;
  if ((in_media_path == nullptr) || (in_media_path[0] == '\0')) { // webcam mode
    cap.open(0);
  } else {
    cap.open(in_media_path); // video loaded from path
  }

  // declare mat vars & input vector
  cv::Mat outImage = cv::Mat::zeros(cv::Size(disp_w, disp_h), CV_32FC3);
  cv::Mat frame, orig_frame, convMat;
  TfLiteTensor *output_mask = nullptr;
  // Allocate tensor buffers
  CHECK_FOR_ERROR(interpreter->AllocateTensors() == kTfLiteOk);

  // diff var to hold inference time in ms
  float diff = 1000.;
  // for removing border feathers
  int kern_size = 1;
  // MORPH_RECT, MORPH_ELLIPSE
  cv::Mat element = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(2 * kern_size + 1, 2 * kern_size + 1),
      cv::Point(kern_size, kern_size));

  // set up opencv video writer if save_path is provided
  cv::VideoWriter video;
  if ((save_path != nullptr) && (save_path[0] != '\0')) {
    int fps = 25;
    video.open(save_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
               cv::Size(disp_w, disp_h));
  }

  // load bg image for replacement
  cv::Mat bg, bg_mask;
  if ((bg_path == nullptr) ||
      (bg_path[0] == '\0')) { // if no bg img path provided, use a zero image
    bg = cv::Mat::zeros(cv::Size(disp_w, disp_h), CV_32FC3);
  } else {
    bg = cv::imread(bg_path); // if bg img path provided, load as BGR
    cv::resize(bg, bg, cv::Size(disp_w, disp_h), cv::INTER_LINEAR);
    bg.convertTo(bg, CV_32FC3);
  }

  // Process video frames till video ends or 'q' is pressed
  while (cv::waitKey(1) != 113) {
    auto start = std::chrono::steady_clock::now();
    // Read frames from camera
    cap >> frame;
    if (frame.empty())
      break;
    orig_frame = frame.clone();
    cv::resize(frame, frame, cv::Size(in_shape.width, in_shape.height),
               cv::INTER_AREA);

    // Scale image to range 0-1 and convert dtype to float 32
    frame.convertTo(frame, CV_32FC3, 1.f / 255);

    // copy image to input as input tensor
    memcpy(interpreter->typed_input_tensor<float>(in_shape.index), frame.data,
           frame.total() * frame.elemSize());

    // Run inference
    interpreter->Invoke();
    // Read output buffers
    output_mask = interpreter->tensor(interpreter->outputs()[out_shape.index]);
    float *output_data_ptr = output_mask->data.f;
    // convert from float* to cv::mat
    cv::Mat two_channel(out_shape.height, out_shape.width, CV_32FC2,
                        output_data_ptr);

    // 0=bg channel, 1=fg channel
    cv::extractChannel(two_channel, convMat, 0);

    // post processing
    // removing edge feathering, smoothen boundaries, resize to disp dims
    cv::dilate(convMat, convMat, element);
    cv::GaussianBlur(convMat, convMat, cv::Size(3, 3), 3);
    cv::resize(convMat, convMat, cv::Size(disp_w, disp_h), cv::INTER_LINEAR);

    cv::threshold(convMat, convMat, settings.threshold, 1.,
                  cv::THRESH_BINARY_INV);
    cv::cvtColor(convMat, convMat, cv::COLOR_GRAY2BGR);

    cv::subtract(bg, convMat * 255.0, bg_mask);
    cv::resize(orig_frame, orig_frame, cv::Size(disp_w, disp_h),
               cv::INTER_LINEAR);
    orig_frame.convertTo(orig_frame, CV_32FC3);

    cv::multiply(convMat, orig_frame, convMat); // mix orig img and mask
    bg_mask.convertTo(bg_mask, CV_8UC3);
    convMat.convertTo(convMat, CV_8UC3);
    cv::add(convMat, bg_mask, convMat);

    cv::String label =
        cv::format("FPS: %.2f: Inference time %.2f ", 1000.0 / diff, diff);
    cv::putText(convMat, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0));
    cv::imshow(disp_window_name, convMat);

    if ((save_path != nullptr) && (save_path[0] != '\0'))
      video.write(convMat);
    auto end = std::chrono::steady_clock::now();
    // Store the time difference between start and end
    diff = std::chrono::duration<double, std::milli>(end - start).count();
  }
  video.release();
  cap.release();
  cv::destroyAllWindows();
  return 0;
}
