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

#include "camera_streamer.hpp"
#include "common.hpp"
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
  Settings settings = get_settings_from_args(argc, argv);

  if (strcmp(settings.mode, "img") == 0) {
    if ((settings.in_media_path == nullptr) ||
        (settings.in_media_path[0] == '\0')) {
      std::cout << "ERROR: Input image path is needed. Use -h for help\n";
      exit(1);
    }
    img_mode(settings);
  } else if (strcmp(settings.mode, "vid") == 0) {
    vid_mode(settings);
  }
  return 0;
}

int img_mode(Settings &settings) {
  const char *model_filepath = settings.model_path;
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
  cv::subtract(bg, convMat * 255.0, bg_mask); // remove person mask from bg

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
  const char *model_filepath = settings.model_path;
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
  std::unique_ptr<CameraStreamer> cap;  // unique_ptr to create class ref without init
  int qsize = 2; // set a small max qsize <= 2
  if ((in_media_path == nullptr) || (in_media_path[0] == '\0')) {
    // webcam mode
    std::vector<int> capture_src = {0};
    cap = std::make_unique<CameraStreamer>(capture_src, qsize);
  } else {
    // video loaded from path
    std::vector<std::string> capture_src = {in_media_path};
    cap = std::make_unique<CameraStreamer>(capture_src, qsize);
  }

  // declare mat vars & input vector
  cv::Mat mask =
      cv::Mat::zeros(cv::Size(out_shape.width, out_shape.height), CV_32FC1);
  cv::Mat frame, orig_frame, convMat;
  TfLiteTensor *output_mask = nullptr;
  // Allocate tensor buffers
  CHECK_FOR_ERROR(interpreter->AllocateTensors() == kTfLiteOk);

  // diff var to hold inference time in ms
  float diff = 1000.;
  // for removing border feathers
  int kern_size = 2;
  // MORPH_RECT, MORPH_ELLIPSE
  cv::Mat element = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(2 * kern_size + 1, 2 * kern_size + 1),
      cv::Point(kern_size, kern_size));

  // set up opencv video writer if save_path is provided
  cv::VideoWriter video;
  if ((save_path != nullptr) && (save_path[0] != '\0')) {
    int fps = 25;  // save FPS
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

  // set prev mask combining params
  bool use_prev_msk = settings.use_prev_msk;
  // 0=bg channel, 1=fg channel
  const int out_channel_index = 0;
  const float combine_with_prev_ratio = 1.f;
  const float eps = 0.001;
  cv::Mat prev_convMat;
  if (use_prev_msk)
    std::cout << "INFO: Using previous masks for stability. Might reduce FPS"
              << '\n';
  // Process video frames till video ends or 'q' is pressed
  while (cv::waitKey(3) != 113) {
    auto start = std::chrono::steady_clock::now();
    // Read frames from camera
    cv::Mat frame;
    if (cap->frame_queue[0]->pop(frame)) {
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
      output_mask =
          interpreter->tensor(interpreter->outputs()[out_shape.index]);
      float *output_data_ptr = output_mask->data.f;
      // convert from float* to cv::mat
      cv::Mat two_channel(out_shape.height, out_shape.width, CV_32FC2,
                          output_data_ptr);

      if (!use_prev_msk) {
        // Use the raw bg mask output, faster but less stable over frames
        cv::extractChannel(two_channel, mask, 0);
      } else {
        // Run softmax over tensor output and blend with previous mask.
        for (int i = 0; i < out_shape.height; ++i) {
          for (int j = 0; j < out_shape.width; ++j) {
            // Only two channel input tensor is supported.
            const cv::Vec2f input_pix = two_channel.at<cv::Vec2f>(i, j);
            const float shift = std::max(input_pix[0], input_pix[1]);
            const float softmax_denom =
                std::exp(input_pix[0] - shift) + std::exp(input_pix[1] - shift);
            float new_mask_value =
                std::exp(input_pix[out_channel_index] - shift) / softmax_denom;

            // Combine prev value with cur using uncertainty^2 as mixing coeff
            if (use_prev_msk && !prev_convMat.empty()) {
              const float prev_mask_value = prev_convMat.at<float>(i, j);
              float uncertainty_alpha =
                  1.0 + (new_mask_value * std::log(new_mask_value + eps) +
                         (1.0 - new_mask_value) *
                             std::log(1.0 - new_mask_value + eps)) /
                            std::log(2.0f);
              uncertainty_alpha = std::clamp(uncertainty_alpha, 0.0f, 1.0f);
              // Equivalent to: a = 1 - (1 - a) * (1 - a);  (uncertainty ^ 2)
              uncertainty_alpha *= 2.0 - uncertainty_alpha;
              const float mixed_mask_value =
                  new_mask_value * uncertainty_alpha +
                  prev_mask_value * (1.0f - uncertainty_alpha);
              new_mask_value =
                  mixed_mask_value * combine_with_prev_ratio +
                  (1.0f - combine_with_prev_ratio) * new_mask_value;
            }
            mask.at<float>(i, j) = new_mask_value;
          }
        }
        prev_convMat = mask.clone();
      }

      // post processing
      // removing edge feathering, smoothen boundaries, resize to disp dims
      cv::dilate(mask, convMat, element);
      cv::GaussianBlur(convMat, convMat, cv::Size(3, 3), 3);
      cv::resize(convMat, convMat, cv::Size(disp_w, disp_h), cv::INTER_LINEAR);

      cv::threshold(convMat, convMat, settings.threshold, 1.,
                    cv::THRESH_BINARY_INV);
      cv::cvtColor(convMat, convMat, cv::COLOR_GRAY2BGR);
      cv::subtract(bg, convMat * 255.0, bg_mask); // remove person mask from bg

      cv::resize(orig_frame, orig_frame, cv::Size(disp_w, disp_h),
                 cv::INTER_LINEAR);
      orig_frame.convertTo(orig_frame, CV_32FC3);

      cv::multiply(convMat, orig_frame, convMat); // mix orig img and mask
      bg_mask.convertTo(bg_mask, CV_8UC3);
      convMat.convertTo(convMat, CV_8UC3);
      cv::add(convMat, bg_mask, convMat);

      cv::String label =
          cv::format("FPS: %.2f: Inference time %.2f ", 1000.0 / diff, diff);
      cv::putText(convMat, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX,
                  0.5, cv::Scalar(0, 255, 0));
      cv::imshow(disp_window_name, convMat);

      if ((save_path != nullptr) && (save_path[0] != '\0'))
        video.write(convMat);
      auto end = std::chrono::steady_clock::now();
      // Store the time difference between start and end
      diff = std::chrono::duration<double, std::milli>(end - start).count();
    }
  }
  video.release();
  cv::destroyAllWindows();
  return 0;
}
