//
// Created by sergio on 16/05/19.
//
#include "../../include/Model.h"
#include "../../include/Tensor.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <unordered_map>
#include <stdlib.h>
#include <stdio.h>
#include <numeric>
#include <chrono>

struct model_info_struct {
        int in_width;            // module input width
        int in_height;           // module input height
        std::string input_name;  // layer input name
        std::string output_name; // layer output name
        std::string model_path;  // model path
} cur_model;

model_info_struct get_model_struct(std::string model_type);
int cam_mode(model_info_struct cur_model_struct);
int img_mode(model_info_struct cur_model_struct, char *img_name);
void print_model_operations(Model model);
std::string get_basename(std::string full_path);

int main(int argc, char *argv[]) {
        // run program like
        // for image:  ./example img m(1/2/3/4) path_to_img
        // for webcam: ./example cam m(1/2/3/4)
        // 1st arg is mode, 2nd arg is model
        if (argc == 4 && strcmp(argv[1], "img") == 0) {
                // load model configuration files
                cur_model = get_model_struct(argv[2]);
                img_mode(cur_model, argv[3]);
                return 0;
        }
        else if (argc == 3 && strcmp(argv[1], "cam") == 0) {
                // load model configuration files
                cur_model = get_model_struct(argv[2]);
                cam_mode(cur_model);
                return 0;
        }
        std::cout << "Invalid number of args. FMT is ./example img m1 <img_path> or ./example cam m1" << std::endl;
        return 1;
}

int img_mode(model_info_struct cur_model_struct, char *img_name) {
        std::cout << "Loading model " << cur_model_struct.model_path << " for img inference on " << img_name << std::endl;
        // Load model settings i.e. path, input dim, & in/out layer names
        Model model(cur_model_struct.model_path);
        int in_width = cur_model_struct.in_width;
        int in_height = cur_model_struct.in_height;
        Tensor input{model, cur_model_struct.input_name}; // input module name
        Tensor output{model,cur_model_struct.output_name}; // output module name

        // uncomment to print contents of model
        // print_model_operations(&model);

        // display window name
        std::string disp_window_name = get_basename(cur_model_struct.model_path);

        // declare mat vars & input vector
        cv::Mat img, inp, convMat, bgcrop;
        std::vector<float> img_data;

        img = cv::imread(img_name, cv::IMREAD_COLOR);
        cv::resize(img, inp, cv::Size(in_width, in_height), cv::INTER_LINEAR);

        // Scale image to range 0-1 and convert dtype to float 32
        inp.convertTo(inp, CV_32FC3, 1.f/255);

        // Put image mat in input vector to be sent into input Tensor
        img_data.assign((float*)inp.data, (float*)inp.data + inp.total() * inp.channels());
        input.set_data(img_data, {1, in_width, in_height, 3});

        model.run({&input}, output); // run model, output is a Tensor with shape 1,128,128,1

        // get model ouput as vector<float> & conv to cv::Mat & resize to 1,128,128
        convMat = cv::Mat(output.get_data<float>()).reshape(1, in_height); // channel, num_rows

        // post processing
        cv::cvtColor(convMat, convMat, cv::COLOR_GRAY2BGR);
        cv::resize(convMat, convMat, cv::Size(1200, 720), cv::INTER_LINEAR);
        cv::resize(img, img, cv::Size(1200, 720), cv::INTER_LINEAR);
        img.convertTo(img, CV_32FC3);
        cv::GaussianBlur(convMat, convMat, cv::Size(3, 3), 3);
        convMat.setTo(1.0, convMat>=0.63);
        convMat.setTo(0.0, convMat<0.63);
        cv::multiply(convMat, img, convMat); // mix orig img and generated mask
        convMat.convertTo(convMat, CV_8UC3);

        cv::imshow(disp_window_name, convMat);
        cv::waitKey(0);
        return 0;
}

int cam_mode(model_info_struct cur_model_struct) {
        std::cout << "Loading model " << cur_model_struct.model_path << " for webcam inference" << std::endl;
        // Load model settings i.e. path, input dim, & in/out layer names
        Model model(cur_model_struct.model_path);
        int in_width = cur_model_struct.in_width;
        int in_height = cur_model_struct.in_height;
        Tensor input{model, cur_model_struct.input_name}; // input module name
        Tensor output{model,cur_model_struct.output_name}; // output module name

        // Create a window for display
        std::string disp_window_name = get_basename(cur_model_struct.model_path);
        namedWindow(disp_window_name, cv::WINDOW_NORMAL);

        // Capture frames from camera
        cv::VideoCapture cap;
        cap.open(0); // Camera

        // declare mat vars & input vector
        cv::Mat frame, orig_frame, convMat;
        std::vector<float> img_data;

        // Process video frames
        while (cv::waitKey(1) < 0)
        {
                auto start = std::chrono::steady_clock::now();
                // Read frames from camera
                cap >> frame;
                if (frame.empty())
                {
                        cv::waitKey();
                        break;
                }
                orig_frame = frame.clone();
                cv::resize(frame, frame, cv::Size(in_width, in_height), cv::INTER_AREA);

                // Scale image to range 0-1 and convert dtype to float 32
                frame.convertTo(frame, CV_32FC3, 1.f/255);

                // Put image mat in input vector to be sent into input Tensor
                img_data.assign((float*)frame.data, (float*)frame.data + frame.total() * frame.channels());
                input.set_data(img_data, {1, in_width, in_height, 3});

                model.run({&input}, output); // run model, output is a Tensor with shape 1,128,128,1

                // get model ouput as vector<float> & conv to cv::Mat & resize to 1,128,128
                convMat = cv::Mat(output.get_data<float>()).reshape(1, in_height); // channel, num_rows

                // post processing
                cv::cvtColor(convMat, convMat, cv::COLOR_GRAY2BGR);
                cv::resize(convMat, convMat, cv::Size(1000, 720), cv::INTER_LINEAR);
                cv::resize(orig_frame, orig_frame, cv::Size(1000, 720), cv::INTER_LINEAR);

                orig_frame.convertTo(orig_frame, CV_32FC3);
                cv::GaussianBlur(convMat, convMat, cv::Size(3, 3), 3);
                convMat.setTo(1.0, convMat>=0.63);
                convMat.setTo(0.0, convMat<0.63);
                cv::multiply(convMat, orig_frame, convMat); // mix orig img and generated mask
                convMat.convertTo(convMat, CV_8UC3);

                auto end = std::chrono::steady_clock::now();
                // Store the time difference between start and end
                auto diff = std::chrono::duration <double, std::milli> (end - start).count();
                cv::String label = cv::format("FPS: %.2f: Inference time %.2f ", 1000.0 / diff, diff);
                cv::putText(convMat, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
                cv::imshow(disp_window_name, convMat);
        }
        return 0;
}

std::string get_basename(std::string full_path) {
        // get model name from full model path
        std::istringstream stream(full_path);
        std::string str;
        while (std::getline(stream, str, '/')) {}
        return str;
}

model_info_struct get_model_struct(std::string model_type) {
        /* available models are m1, m2, m3, m4, m5, m6 discussed below */
        model_info_struct chosen_model;

        int input_width, input_height;
        std::string input_layer, output_layer, model_path;

        const static std::unordered_map<std::string,int> model_num_map{
                {"m1",1},
                {"m2",2},
                {"m3",3},
                {"m4",4}
        };

        switch(model_num_map.count(model_type) ? model_num_map.at(model_type) : 0) {
        case 1:
                std::cout << "Model mnetv2_munet_transpose_e260_s128 chosen" << std::endl;
                input_width = 128;
                input_height = 128;
                input_layer = "input_1";
                output_layer = "op/Sigmoid";
                model_path = "../model_zoo/mnetv2_munet_transpose_e260_s128.pb";
                break;
        case 2:
                std::cout << "Model mnetv2_munet_transpose_orig_s128 chosen" << std::endl;
                input_width = 128;
                input_height = 128;
                input_layer = "input_1";
                output_layer = "op/Sigmoid";
                model_path = "../model_zoo/mnetv2_munet_transpose_orig_s128.pb";
                break;
        case 3:
                std::cout << "Model mnetv2_munet_bilinear_orig_s128 chosen" << std::endl;
                input_width = 128;
                input_height = 128;
                input_layer = "input_1";
                output_layer = "op/Sigmoid";
                model_path = "../model_zoo/mnetv2_munet_bilinear_orig_s128.pb";
                break;
        case 4:
                std::cout << "Model prisma_orig_s256 chosen" << std::endl;
                input_width = 256;
                input_height = 256;
                input_layer = "input_3";
                output_layer = "op/Sigmoid";
                model_path = "../model_zoo/prisma_orig_s256.pb";
                break;
        case 0:   // for the undefined case
                std::cout << "Error. Model type " << model_type << " not available" << std::endl;
                exit(1);
        }
        chosen_model.in_width = input_width;
        chosen_model.in_height = input_height;
        chosen_model.input_name = input_layer;
        chosen_model.output_name = output_layer;
        chosen_model.model_path = model_path;
        return chosen_model;
}

void print_model_operations(Model *model) {
        // prints all the operations available for a model
        // call function as print_model_operations(&model);
        std::vector<std::string> model_ops = model->get_operations();
        /* to print the contents of the vector result */
        for (std::vector<std::string>::iterator t=model_ops.begin(); t!=model_ops.end(); ++t)
        {
                std::cout<<*t<<std::endl;
        }
}
