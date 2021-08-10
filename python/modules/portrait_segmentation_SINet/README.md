# SINet: Extreme Lightweight Portrait Segmentation

This repository is a CPU inference-only implementation based on SINet ([paper](https://arxiv.org/abs/1911.09099)) and their official GitHub [code implementation](https://github.com/clovaai/ext_portrait_segmentation).

## Build for distribution

If you want to create a standalone SINet Python Package with dependencies included for offline installation, run:

```shell
$ python setup.py package
```

The `SINet-0.1.tar.gz` archive is created under the `dist` folder.

## Offline installation

Setup a virtualenv before proceeding. The tar.gz should be unzipped and the requirements be installed offline with pip using:

```shell
$ pip install -r requirements.txt --no-index --find-links wheelhouse
```

To install the `hello` package from inside the unzipped root folder, run:

```shell
$ python setup.py install # or pip install .
```

### Remove build artifacts:

In Linux/Darwin systems, run:

```shell
$ bash cleanup.sh
```

In Windows systems, run:

```shell
$ cleanup.bat
```

### Note:

The wheelhouse package contains compiled wheel files so the OS and python versions matter. (i.e. dist from Windows system is not compatible with Darwin/Linux systems)

## Requirements

### Testing

-   python3.7
-   torch===1.4.0
-   torchvision===0.5.0
-   opencv-python
-   numpy==1.19.3
-   visdom

## Model

This code is based on SINet ([paper](https://arxiv.org/abs/1911.09099)) Accepted in WACV2020

Hyojin Park, Lars Lowe Sjösund, YoungJoon Yoo, Nicolas Monet, Jihwan Bang, Nojun Kwak

SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze Modules and Information Blocking Decoder

-   config file : SINet.json
-   Param : 0.087 M
-   Flop : 0.064 G
-   IoU : 95.2

## Run example

1.  single image

```shell
$ python Test_image.py
```

2.  single video

```shell
$ python Test_image.py
```

3.  webcam

```shell
$ python Test_webcam.py
```

## Additional Dataset

Augmented dataset is made from Baidu fashion dataset.

The original Baidu dataset link is [here](http://www.cbsr.ia.ac.cn/users/ynyu/dataset/)

EG1800 dataset link in [here](https://drive.google.com/file/d/18xM3jU2dSp1DiDqEM6PVXattNMZvsX4z/view?usp=sharing)

Original author's augmented dataset at [here](https://drive.google.com/file/d/1zkh7gAhWwoX1nR5GzTzBziG8tgTKtr73/view?usp=sharing).
All train and val datasets are used for training segmentation model.

## CityScape

SINet code for cityscapes dataset present at this [link](https://github.com/clovaai/c3_sinet).

## Citation

```shell
@article{park2019extremec3net,
  title={ExtremeC3Net: Extreme Lightweight Portrait Segmentation Networks using Advanced C3-modules},
  author={Park, Hyojin and Sj{\"o}sund, Lars Lowe and Yoo, YoungJoon and Kwak, Nojun},
  journal={arXiv preprint arXiv:1908.03093},
  year={2019}
}

@article{park2019sinet,
  title={SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze Modules and Information Blocking Decoder},
  author={Park, Hyojin and Sj{\"o}sund, Lars Lowe and Monet, Nicolas and Yoo, YoungJoon and Kwak, Nojun},
  journal={arXiv preprint arXiv:1911.09099},
  year={2019}
}
```

## Acknowledgements

-   [Clova AI, NAVER](https://github.com/clovaai)
-   Hyojin Park, Lars Lowe Sjösund and YoungJoon Yoo from  [Clova AI, NAVER](https://clova.ai/en/research/research-areas.html),
    Nicolas Monet from [NAVER LABS Europe](https://europe.naverlabs.com/)
    and Jihwan Bang from [Search Solutions, Inc](https://www.searchsolutions.co.kr/)

## License

    Copyright (c) 2019-present NAVER Corp.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
