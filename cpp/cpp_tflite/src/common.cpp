#include "common.h"
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>

std::string get_basename(std::string full_path) {
  // get model name from full model path
  std::istringstream stream(full_path);
  std::string str;
  while (std::getline(stream, str, '/')) {
  }
  return str;
}

Settings get_settings(std::string model_type) {
  Settings s;
  std::string model_path;
  const static std::unordered_map<std::string, int> model_num_map{{"m1", 1}};

  switch (model_num_map.count(model_type) ? model_num_map.at(model_type) : 0) {
  case 1:
    std::cout << "Selfie Seg Model 256x144 chosen" << std::endl;
    model_path = "../model_zoo/144x256_float32.tflite";
    break;
  case 0: // for the undefined case
    std::cout << "Error. Model type " << model_type << " not available"
              << std::endl;
    exit(1);
  }
  s.model_path = model_path;
  return s;
}
