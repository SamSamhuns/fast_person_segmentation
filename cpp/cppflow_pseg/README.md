# CppFlow

Run TensorFlow models in c++ without Bazel, without TensorFlow installation and without compiling Tensorflow.

```c++
    // Read the graph
    Model model{"graph.pb"};
    model.init();

    // Prepare inputs and outputs
    Tensor input{model, "input"};
    Tensor output{model, "output"};

    // Run
    model.run(input, output);
```

CppFlow uses Tensorflow C API to run the models, meaning you can use it without installing Tensorflow and without compiling the whole TensorFlow repository with bazel, you just need to download the C API. With this project you can manage and run your models in C++ without worrying about _void, malloc or free_. With CppFlow you easily can:

-   Open .pb models created with Python
-   Restore checkpoints
-   Save new checkpoints
-   Feed new data to your inputs
-   Retrieve data from the outputs

## How To Run It

Requires:

-   Tensorflow C/C++ API

-   Opencv C/C++ API

Download [TensorFlow C API](https://www.tensorflow.org/install/lang_c) and put it in the home directory i.e. `/Users/username`. Install the opencv c++ library as well (Using homebrew in OSX or alter path in `CMakeList.txt` for windows).

You can either install the library system wide by following the tutorial on the Tensorflow page or you can place the contents of the archive
in a folder called `libtensorflow` in the home directory.

Afterwards, you can run the examples:

```sh
cd cppflow_pseg/examples/pseg_model
mkdir build
cd build
cmake ..
make
# image segmentation & saving in save_path
./example -m img -t m1 -i <img_path> -s <save_path> # model types are m(1/2/3/4/5/6)
# video segmentation with opencv  & saving in save_path
./example -m vid -t m1 -i <vid_path> -s <save_path> # model types are m(1/2/3/4/5/6)     
# webcam segmentation with opencv & saving in save_path
./example -m vid -t m1 -s <save_path>               # model types are m(1/2/3/4/5/6)
# webcam segmentation with opencv  without saving
./example -m vid -t m1                              # model types are m(1/2/3/4/5/6)
```

## Usage

Suppose we have a saved graph defined by the following TensorFlow Python code (_examples/pseg_model/create_model.py_):

```Python
# Two simple inputs
a = tf.placeholder(tf.float32, shape=(1, 100), name="input_a")
b = tf.placeholder(tf.float32, shape=(1, 100), name="input_b")
# Output
c = tf.add(a, b, name='result')
```

### Create Model

You need the graph definition in a .pb file to create a model (_examples/pseg_model/model.pb_), then you can init it or restore from checkpoint

```c++
Model model("graph.pb");
// Initialize the variables...
model.init();
// ... or restore from checkpoint
model.restore("train.ckpt")
```

### Define Inputs and Outputs

You can create the Tensors by the name of the operations (if you don't know use model.get_operations())

```c++
Tensor input_a{model, "input_a"};
Tensor input_b{model, "input_b"};
Tensor output{model, "result"};
```

### Feed new data to the inputs

Excepected inputs have a shape=(1,100), therefore we have to supply 100 elements:

```c++
// Create a vector data = {0,1,2,...,99}
std::vector<float> data(100);
std::iota(data.begin(), data.end(), 0);

// Feed data to the inputs
input_a->set_data(data);
input_b->set_data(data);
```

### Run and get the ouputs

```c++
// Run!
model.run({input_a, input_b}, output);

// Write the output: 0, 2, 4, 6,.., 198
for (float f : output->get_data<float>()) {
    std::cout << f << " ";
}
std::cout << std::endl;
```
