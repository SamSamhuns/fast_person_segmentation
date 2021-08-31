#include "../../include/Model.h"
#include "../../include/Tensor.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <getopt.h>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>

struct model_info_struct {
  float threshold;         // model threshold
  int in_width;            // module input width
  int in_height;           // module input height
  std::string input_name;  // layer input name
  std::string output_name; // layer output name
  std::string model_path;  // model path
  char *in_media_path;     // input media path img/vid
  char *save_path;         // inferenced img/vid save
  bool verbose;            // verbose mode
} mS;

model_info_struct get_model_struct(std::string model_type, char *in_media_path,
                                   char *save_path, bool verbose);
int img_mode(model_info_struct m_struct);
int vid_mode(model_info_struct m_struct);
void print_model_operations(Model model);
std::string get_basename(std::string full_path);
std::tuple<char *, char *, char *, char *, char *, bool>
parse_args(int argc, char **argv);

int main(int argc, char *argv[]) {
  char *mode, *model_type, *in_media_path, *bg_path, *save_path;
  bool verbose;
  std::tie(mode, model_type, in_media_path, bg_path, save_path, verbose) =
      parse_args(argc, argv);

  mS = get_model_struct(model_type, in_media_path, save_path, verbose);
  if (strcmp(mode, "img") == 0) {
    if ((in_media_path == nullptr) || (in_media_path[0] == '\0')) {
      std::cout << "ERROR: Input image path is needed. Use -h for help\n";
      exit(1);
    }
    img_mode(mS);
  } else if (strcmp(mode, "vid") == 0) {
    vid_mode(mS);
  }
  return 0;
}

int img_mode(model_info_struct m_struct) {
  // Load model settings i.e. path, input dim, & in/out layer names
  Model model(m_struct.model_path);
  int in_width = m_struct.in_width;
  int in_height = m_struct.in_height;
  Tensor input{model, m_struct.input_name};   // input module name
  Tensor output{model, m_struct.output_name}; // output module name
  float threshold = m_struct.threshold;
  char *in_media_path = m_struct.in_media_path;
  char *save_path = m_struct.save_path;

  // uncomment to print contents of model
  // print_model_operations(&model);

  // display window name
  std::string disp_window_name = get_basename(m_struct.model_path);

  // declare mat vars & input vector
  cv::Mat img, inp, convMat, bgcrop;
  std::vector<float> img_data;

  img = cv::imread(in_media_path, cv::IMREAD_COLOR);
  cv::resize(img, inp, cv::Size(in_width, in_height), cv::INTER_LINEAR);

  // Scale image to range 0-1 and convert dtype to float 32
  inp.convertTo(inp, CV_32FC3, 1.f / 255);

  // Put image mat in input vector to be sent into input Tensor
  img_data.assign((float *)inp.data,
                  (float *)inp.data + inp.total() * inp.channels());
  input.set_data(img_data, {1, in_width, in_height, 3});

  model.run({&input},
            output); // run model, output is a Tensor with shape 1,128,128,1

  // get model ouput as vector<float> & conv to cv::Mat & resize to 1,128,128
  convMat = cv::Mat(output.get_data<float>())
                .reshape(1, in_height); // channel, num_rows

  // post processing
  cv::cvtColor(convMat, convMat, cv::COLOR_GRAY2BGR);
  cv::resize(convMat, convMat, cv::Size(1200, 720), cv::INTER_LINEAR);
  cv::resize(img, img, cv::Size(1200, 720), cv::INTER_LINEAR);
  img.convertTo(img, CV_32FC3);
  cv::GaussianBlur(convMat, convMat, cv::Size(3, 3), 3);
  convMat.setTo(1.0, convMat >= threshold);
  convMat.setTo(0.0, convMat < threshold);
  cv::multiply(convMat, img, convMat); // mix orig img and generated mask
  convMat.convertTo(convMat, CV_8UC3);

  cv::imshow(disp_window_name, convMat);
  // save output save_path if provided
  if ((save_path != nullptr) && (save_path[0] != '\0'))
    cv::imwrite(save_path, convMat);
  cv::waitKey(0);
  return 0;
}

int vid_mode(model_info_struct m_struct) {
  // Load model settings i.e. path, input dim, & in/out layer names
  Model model(m_struct.model_path);
  int in_width = m_struct.in_width;
  int in_height = m_struct.in_height;
  Tensor input{model, m_struct.input_name};   // input module name
  Tensor output{model, m_struct.output_name}; // output module name
  float threshold = m_struct.threshold;
  char *in_media_path = m_struct.in_media_path;
  char *save_path = m_struct.save_path;

  // Create a window for display
  std::string disp_window_name = get_basename(m_struct.model_path);
  namedWindow(disp_window_name, cv::WINDOW_NORMAL);

  // Capture frames from camera
  cv::VideoCapture cap;
  if ((in_media_path == nullptr) || (in_media_path[0] == '\0')) { // webcam mode
    cap.open(0);
  } else {
    cap.open(in_media_path); // video loaded from path
  }

  // declare mat vars & input vector
  cv::Mat frame, orig_frame, convMat;
  std::vector<float> img_data;

  // diff var to hold inference time in ms
  float diff = 1000.;

  // set up opencv video writer if save_path is provided
  cv::VideoWriter video;
  if ((save_path != nullptr) && (save_path[0] != '\0')) {
    int fps = 25;
    video.open(save_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
               cv::Size(1000, 720));
  }

  // Process video frames till video ends or 'q' is pressed
  while (cv::waitKey(1) != 113) {
    auto start = std::chrono::steady_clock::now();
    // Read frames from camera
    cap >> frame;
    if (frame.empty())
      break;
    orig_frame = frame.clone();
    cv::resize(frame, frame, cv::Size(in_width, in_height), cv::INTER_AREA);

    // Scale image to range 0-1 and convert dtype to float 32
    frame.convertTo(frame, CV_32FC3, 1.f / 255);

    // Put image mat in input vector to be sent into input Tensor
    img_data.assign((float *)frame.data,
                    (float *)frame.data + frame.total() * frame.channels());
    input.set_data(img_data, {1, in_width, in_height, 3});

    model.run({&input},
              output); // run model, output is a Tensor with shape 1,128,128,1

    // get model ouput as vector<float> & conv to cv::Mat & resize to 1,128,128
    convMat = cv::Mat(output.get_data<float>())
                  .reshape(1, in_height); // channel, num_rows

    // post processing
    cv::cvtColor(convMat, convMat, cv::COLOR_GRAY2BGR);
    cv::resize(convMat, convMat, cv::Size(1000, 720), cv::INTER_LINEAR);
    cv::resize(orig_frame, orig_frame, cv::Size(1000, 720), cv::INTER_LINEAR);

    orig_frame.convertTo(orig_frame, CV_32FC3);
    cv::GaussianBlur(convMat, convMat, cv::Size(3, 3), 3);
    convMat.setTo(1.0, convMat >= threshold);
    convMat.setTo(0.0, convMat < threshold);
    cv::multiply(convMat, orig_frame,
                 convMat); // mix orig img and generated mask
    convMat.convertTo(convMat, CV_8UC3);

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

std::string get_basename(std::string full_path) {
  // get model name from full model path
  std::istringstream stream(full_path);
  std::string str;
  while (std::getline(stream, str, '/')) {
  }
  return str;
}

model_info_struct get_model_struct(std::string model_type, char *in_media_path,
                                   char *save_path, bool verbose) {
  /* available models are m1, m2, m3, m4, m5, m6 discussed below */
  model_info_struct chosen_model;
  float threshold;
  int input_width, input_height;
  std::string input_layer, output_layer, model_path;

  const static std::unordered_map<std::string, int> model_num_map{
      {"m1", 1}, {"m2", 2}, {"m3", 3}, {"m4", 4}, {"m5", 4}};

  switch (model_num_map.count(model_type) ? model_num_map.at(model_type) : 0) {
  case 1:
    std::cout << "Model mnetv2_munet_transpose_e260_s128 chosen" << std::endl;
    threshold = 0.63;
    input_width = 128;
    input_height = 128;
    input_layer = "input_1";
    output_layer = "op/Sigmoid";
    model_path = "../model_zoo/mnetv2_munet_transpose_e260_s128.pb";
    break;
  case 2:
    std::cout << "Model mnetv2_munet_transpose_orig_s128 chosen" << std::endl;
    threshold = 0.63;
    input_width = 128;
    input_height = 128;
    input_layer = "input_1";
    output_layer = "op/Sigmoid";
    model_path = "../model_zoo/mnetv2_munet_transpose_orig_s128.pb";
    break;
  case 3:
    std::cout << "Model mnetv2_munet_bilinear_orig_s128 chosen" << std::endl;
    threshold = 0.63;
    input_width = 128;
    input_height = 128;
    input_layer = "input_1";
    output_layer = "op/Sigmoid";
    model_path = "../model_zoo/mnetv2_munet_bilinear_orig_s128.pb";
    break;
  case 4:
    std::cout << "Model prisma_orig_s256 chosen" << std::endl;
    threshold = 0.63;
    input_width = 256;
    input_height = 256;
    input_layer = "input_3";
    output_layer = "op/Sigmoid";
    model_path = "../model_zoo/prisma_orig_s256.pb";
    break;
  case 5:
    std::cout << "Model mnetv3_decoder chosen" << std::endl;
    threshold = 0.8;
    input_width = 256;
    input_height = 144;
    input_layer = "input_1";
    output_layer = "segment";
    model_path = "../model_zoo/mnetv3_decoder_144x256.pb";
    break;
  case 0: // for the undefined case
    std::cout << "Error. Model type " << model_type << " not available"
              << std::endl;
    exit(1);
  }
  chosen_model.in_width = input_width;
  chosen_model.in_height = input_height;
  chosen_model.input_name = input_layer;
  chosen_model.output_name = output_layer;
  chosen_model.model_path = model_path;
  chosen_model.threshold = threshold;
  chosen_model.in_media_path = in_media_path;
  chosen_model.save_path = save_path;
  chosen_model.verbose = verbose;
  return chosen_model;
}

void print_model_operations(Model *model) {
  // prints all the operations available for a model
  // call function as print_model_operations(&model);
  std::vector<std::string> model_ops = model->get_operations();
  /* to print the contents of the vector result */
  for (std::vector<std::string>::iterator t = model_ops.begin();
       t != model_ops.end(); ++t) {
    std::cout << *t << std::endl;
  }
}

void print_help() {
  std::cout
      << "--mode/-m <img/vid>:           Use -m img/vid for image/video mode\n"
         "--model_type/-t <path>:        Model type <m1/m2/m3/m4/m5>\n"
         "--in_media_path/-i <path>:     Path to fg img/vid given mode\n"
         "--bg_image_path/-b <path>:     Path to bg image."
         "If absent, use a black background\n"
         "--save_path/-s <path>:         Path to save inference image/video."
         "If absent, no results saved\n"
         "--verbose/--v:                 Verbose mode if flag specified\n"
         "--help/-h:                     Show help\n";
}

std::tuple<char *, char *, char *, char *, char *, bool>
parse_args(int argc, char **argv) {
  // init return char ptr vars to nullptr for initialization check later
  char *mode = nullptr;
  char *model_type = nullptr;
  char *in_media_path = nullptr;
  char *bg_image_path = nullptr;
  char *save_path = nullptr;
  bool verbose = false;

  const char *const short_opts = "m:t:i:b:s:h";
  const option long_opts[] = {
      {"mode", required_argument, nullptr, 'm'},
      {"model_type", required_argument, nullptr, 't'},
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
      model_type = optarg;
      std::cout << "Model type: " << model_type << "\n";
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

  // mode and model_type must not be nullptr
  // note for this check to happen, vars must be initialized to nullptr
  if ((mode == nullptr) || (mode[0] == '\0')) {
    std::cout << "ERROR: -m img/vid must be specified. Use -h for help" << '\n';
    exit(-1);
  } else if ((model_type == nullptr) || (model_type[0] == '\0')) {
    std::cout << "ERROR: model type is needed. Use -h for help \n";
    exit(-1);
  }
  return std::make_tuple(mode, model_type, in_media_path, bg_image_path,
                         save_path, verbose);
}
