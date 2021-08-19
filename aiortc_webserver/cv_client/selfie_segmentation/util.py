# utils for model inference
from pathlib import Path, PurePath
from threading import Thread
from time import sleep
from enum import Enum
import numpy as np
import subprocess
import imageio
import cv2
import sys


class PostProcessingType(Enum):
    """Post_Processing methods
    """
    GAUSSIAN = "gaussian"
    MORPH_OPEN = "morph_open"


class VideoStreamMT(object):
    def __init__(self, src=0):
        print(f"INFO: Setting up multi-threading video IO from src {src}")
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


class ImageioVideoWriter(object):

    def __init__(self, output_dir, video_name, model_fname, fps=25):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        video_name = Path(model_fname).stem + Path(video_name).stem + ".mp4"
        video_save_path = str(PurePath(output_dir, video_name))

        print(f"INFO: Output video will be saved in {video_save_path}")
        self.writer = imageio.get_writer(video_save_path, fps=fps)

    def write_frame(self, image):
        if isinstance(image, str):
            self.writer.append_data(imageio.imread(image))
        else:
            self.writer.append_data(image)

    def close(self):
        self.writer.close()


def install_pip_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def _default_bg_load_transform(img):
    """ Convert to RGB space and normlize to [0,1] range """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0


def load_bgd(bg_img_path, bg_w, bg_h, dtype=np.float32, post_process=_default_bg_load_transform):
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


def get_frame_after_postprocess(msk, img, bgd, bg_wh, disp_wh, threshold, foreground="img"):
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
