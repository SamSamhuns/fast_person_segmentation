# utils for model inference
from typing import Tuple, Union, Callable
from pathlib import Path, PurePath
from threading import Thread
from queue import Queue
from time import sleep
from enum import Enum
import argparse
import json

import numpy as np
import imageio
import cv2


class BackgroundMode(Enum):
    """Background Mode
    """
    BLUR = "blur"


class PostProcessingType(Enum):
    """Post_Processing methods
    """
    GAUSSIAN = "gaussian"
    MORPH_OPEN = "morph_open"


class VideoStreamMTQueueWidget(object):
    def __init__(self, src: Union[int, str] = 0, maxsize: int = 3):
        """
        Does not allow dropping of frames with the use of blocking queues
        Args:
            src: path to video or 0 for webcam or other camera index
            maxsize: size of frame load Queue
        """
        print(
            f"INFO: Setting up blocking queue multi-threading video IO from src {src}")
        self.capture = cv2.VideoCapture(src)
        self.frame_queue = Queue(maxsize=maxsize)
        # immediately start a thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                status, frame = self.capture.read()
                # blocking operation if queue is full
                self.frame_queue.put((status, frame))
            else:
                return

    def read(self):
        return self.frame_queue.get()

    def release(self):
        self.capture.release()

    def get(self, code: int):
        return self.capture.get(code)


class VideoStreamMTNoQueueWidget(object):
    def __init__(self, src: Union[int, str] = 0):
        """
        Recommended for webcam use
        This class is faster than VideoStreamMTQueueWidget
        Queue blocking is not used for storing frames so webcam frames might be dropped
        Args:
            src: path to video or 0 for webcam or other camera index
        """
        print(
            f"INFO: Setting up no-queue multi-threading video IO from src {src}")
        self.capture = cv2.VideoCapture(src)
        self.status = self.capture.isOpened()
        # immediately start a thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.frame = None

    def update(self):
        # read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            else:
                return
            sleep(.005)

    def read(self):
        return self.status, self.frame

    def release(self):
        self.capture.release()

    def get(self, code: int):
        return self.capture.get(code)


class ImageioVideoWriter(object):

    def __init__(self, output_dir: str, video_name: str, model_fname: str, fps: int = 25):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        video_name = Path(model_fname).stem + Path(video_name).stem + ".mp4"
        video_save_path = str(PurePath(output_dir, video_name))

        print(f"INFO: Output video will be saved in {video_save_path}")
        self.writer = imageio.get_writer(video_save_path, fps=fps)

    def write_frame(self, image: Union[str, np.ndarray]):
        if isinstance(image, str):
            self.writer.append_data(imageio.imread(image))
        else:
            self.writer.append_data(image)

    def close(self):
        self.writer.close()


def get_video_stream_widget(vid_path: str):
    """
    returns a video stream widget based on the video_src
    """
    if isinstance(vid_path, int):
        return VideoStreamMTNoQueueWidget(vid_path)
    elif isinstance(vid_path, str) and Path(vid_path).is_file():
        return VideoStreamMTQueueWidget(vid_path)


def get_cmd_argparser(default_model="models/transpose_seg/deconv_bnoptimized_munet_e260.hdf5"):
    """
    get a argparse.ArgumentParser object, must run parser.parse_args() to parse cmd args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--source_vid_path',
                        type=str,
                        help="""Source video on which person seg is run.
                        The default webcam is used is no path provided""")
    parser.add_argument('-b', '--bg_img_path',
                        type=str,
                        help="""Path to image which will replace the background.
                                A dark background is used is path is not provided""")
    parser.add_argument('-m', '--model_path',
                        type=str,
                        default=default_model,
                        help="Path to inference model (default: %(default)s)")
    parser.add_argument("--ds", "--disp_wh_size", dest="disp_wh_size",
                        nargs=2, default=(1280, 720),
                        help="Displayed frames are resized to this (width, height). (default: %(default)s).")
    parser.add_argument('--mt', '--use_multi_thread',
                        dest="use_multi_thread",
                        action="store_true",
                        help="Flag to use multi_thread for opencv video io. (default: %(default)s)")
    parser.add_argument('-o', '--output_dir',
                        default=None,
                        help="Dir where inferenced video will be saved if path is not None")
    return parser


def remove_argparse_option(parser, arg):
    """
    args:
        parser: parser object
        arg: argument name without leading dash
    """
    for action in parser._actions:
        if (vars(action)['option_strings']
            and vars(action)['option_strings'][0] == arg) \
                or vars(action)['dest'] == arg:
            parser._remove_action(action)

    for action in parser._action_groups:
        vars_action = vars(action)
        var_group_actions = vars_action['_group_actions']
        for x in var_group_actions:
            if x.dest == arg:
                var_group_actions.remove(x)
                return


def get_config_dict(model_path: str, json_config_path: str) -> dict:
    """
    load json_config_path as dict and return config of selected model
        model_path: path to h5/pb/tflite model file
        json_config_path: model config json file
    """
    config_dict = {}
    with open(json_config_path) as cfile:
        config_dict = json.load(cfile)
        config_dict = config_dict[Path(model_path).stem]

    return config_dict


def _default_bg_load_transform(img: np.ndarray) -> np.ndarray:
    """ Convert to RGB space and normlize to [0,1] range """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0


def load_bgd(
        bg_img_path: str,
        bg_w: int,
        bg_h: int,
        dtype=np.float32,
        post_process: Callable = _default_bg_load_transform) -> np.ndarray:
    """
    loads & preprocesses bg img
    if bg_img_path is None, return a black image image
    """
    if bg_img_path is None:
        bgd = np.zeros([bg_h, bg_w, 3], dtype=dtype)
    elif isinstance(bg_img_path, str) and Path(bg_img_path).is_file():
        bgd = cv2.resize(cv2.imread(bg_img_path),
                         (bg_w, bg_h)).astype(dtype)
        if post_process is not None:
            bgd = post_process(bgd)
    else:
        raise Exception(f"{bg_img_path} not a path to image")
    return bgd


def get_frame_after_postprocess(
        msk: np.ndarray,
        img: np.ndarray,
        bgd: np.ndarray,
        bg_wh: Tuple[int, int],
        disp_wh: Tuple[int, int],
        threshold: float,
        foreground: str = "img") -> np.ndarray:
    bg_w, bg_h = bg_wh
    disp_w, disp_h = disp_wh
    # resize mask to bg size and apply thres
    msk = cv2.resize(msk, (bg_w, bg_h)).reshape(
        (bg_h, bg_w, 1)) > threshold
    img = cv2.resize(img, (bg_w, bg_h))
    # alpha blending: frame = (img * msk) + (bgd * (1 - msk))
    if foreground == "img":
        frame = np.where(msk, img, bgd).astype(np.uint8)
    elif foreground == "bgd":
        frame = np.where(msk, bgd, img).astype(np.uint8)
    # resize to final resolution
    frame = cv2.resize(frame, (disp_w, disp_h))
    return frame
