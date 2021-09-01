# Building tensorflow-lite cpp

[Official source build instructions](https://www.tensorflow.org/lite/guide/build_cmake)

```shell
$ git clone https://github.com/tensorflow/tensorflow.git
```

For using tflite inference, the include dir must have the following structure:

    └── include
        ├── common
        │   └── common.h
        ├── tensorflow
        │   ├── core
        │   │   ├── kernels
        │   │   ├── public
        │   │   └── util
        │   └── lite
        └── third_party
            ├── eigen3
            └── fft2d

These files can be coped from the original tensorflow git repo.

Install opencv c++ library. For OSX, use homebrew, `brew install opencv`. For Linux follow instructions [here](https://docs.opencv.org/4.5.0/d7/d9f/tutorial_linux_install.html)

## tflite inference setup

```shell
$ mkdir build; cd build;
$ cmake ..
$ cmake --build .
$ ./tflite_seg img m1 <PATH_TO_IMG>
$ ./tflite_seg cam m1
```

## Running Inference

```shell
$ ./tflite_seg -m img -t <model_path> -i <in_image_path> -b <bg_img_path> -s <save_path>
$ ./tflite_seg -m vid -t <model_path> -i <in_video_path> -b <bg_img_path> -s <save_path>
$ ./tflite_seg --help
```

## Building tensorflow python with XNNPACK support

[Official source build instructions](https://www.tensorflow.org/install/source)

While building in mac-osx, if there is an error related to google protobufs, try:

```shell
# Verify that nothing is using the installed formula:
$ brew uses protobuf --installed
# Remove the protobuf formula from your system:
$ brew uninstall protobuf
```

## Git clone tensorflow repo and setup a virtual env and install dependencies:

```shell
$ git clone https://github.com/tensorflow/tensorflow.git; cd tensorflow
$ python -m venv venv;source venv
$ pip install numpy wheel
$ pip install keras_preprocessing --no-deps
```

## Configure and start build

Notes: `--copt=-mfpmath=both` is incompatible with clang

```shell
$ ./configure
$ bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --define tflite_with_xnnpack=true //tensorflow/tools/pip_package:build_pip_package
$ ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
$ pip install /tmp/tensorflow_pkg/tensorflow-<version>-<tags>.whl
```

If there is an issue with numpy, upgrade with pip: `pip install numpy --upgrade`

## Building tensorflowlite cpp package with bazel only

```shell
$ bazel build -c opt --define tflite_with_xnnpack=true //tensorflow/lite:libtensorflowlite.dylib
```
