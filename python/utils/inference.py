# utils for model inference
from threading import Thread
from os import makedirs
from time import sleep
import os.path as osp
import numpy as np
import argparse
import imageio
import json
import cv2


class VideoStreamMultiThreadWidget(object):
    def __init__(self, src=0):
        print(f"INFO: Setting up multi-threading video IO from src {src}")
        self.capture = cv2.VideoCapture(src)
        # start a thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.frame = None

    def update(self):
        # read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            sleep(.01)

    def show_frame(self):
        cv2.imshow('frame', self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

    def read(self):
        return self.capture.isOpened(), self.frame


class ImageioVideoWriter(object):

    def __init__(self, output_save_path, video_name, fps=25):
        makedirs(output_save_path, exist_ok=True)
        video_name = osp.basename(video_name).split('.')[0] + ".mp4"
        video_save_path = osp.join(output_save_path, video_name)
        print(f"INFO: Output video will be saved in {video_save_path}")
        self.writer = imageio.get_writer(video_save_path, fps=fps)

    def write_frame(self, image):
        if isinstance(image, str):
            self.writer.append_data(imageio.imread(image))
        else:
            self.writer.append_data(image)

    def close(self):
        self.writer.close()


def get_cmd_argparser(default_model="models/transpose_seg/deconv_bnoptimized_munet_e260.hdf5"):
    """
    get a argparse.ArgumentParser object, must run parser.parse_args() to parse cmd args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--source_vid_path',
                        type=str,
                        required=False,
                        help="""Source video on which person seg is run.
                        The default webcam is used is no path provided""")
    parser.add_argument('-b',
                        '--bg_img_path',
                        type=str,
                        required=False,
                        help="""Path to image which will replace the background.
                                A dark background is used is path is not provided""")
    parser.add_argument('-m',
                        '--model_path',
                        type=str,
                        required=False,
                        default=default_model,
                        help="Path to inference model (i.e. h5/tflite/pb fmt)")
    parser.add_argument('-mt',
                        '--use_multi_thread',
                        action="store_true",
                        required=False,
                        help="Flag to use multi_thread for opencv video io. Default is to use single thread")
    parser.add_argument('-o',
                        '--output_save_path',
                        default=None,
                        help="Path to dir where inferenced video will be saved if path is not None")
    return parser


def get_config_dict(model_path, json_config_path):
    """
    load json_config_path as dict and return config of selected model
        model_path: path to h5/pb/tflite model file
        json_config_path: model config json file
    """
    config_dict = {}
    with open(json_config_path) as cfile:
        config_dict = json.load(cfile)
        config_dict = config_dict[osp.basename(model_path).split('.')[0]]

    return config_dict


def _default_post_process(img):
    """ Convert to RGB space and normlize to [0,1] range """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0


def load_bgd(bg_img_path, bg_w, bg_h, dtype=np.float32, post_process=_default_post_process):
    """
    loads & preprocesses bg img
    if bg_img_path is None, return a black image image
    """
    if bg_img_path is None:
        bgd = np.zeros([bg_h, bg_w, 3], dtype=dtype)
    elif isinstance(bg_img_path, str) and osp.isfile(bg_img_path):
        bgd = cv2.resize(cv2.imread(bg_img_path),
                         (bg_w, bg_h)).astype(dtype)
        if post_process is not None:
            bgd = post_process(bgd)
    else:
        raise Exception(f"{bg_img_path} not a path to image")
    return bgd
