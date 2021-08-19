import cv2
import numpy as np
import tensorflow as tf
from .util import load_bgd, get_frame_after_postprocess


class SelfieSegmentation(object):

    def __init__(self, bg_img_path=None, tflite_model_path="weights/model_float16_quant.tflite"):
        """
        bg_img_path: numpy.ndarray, if set to None, a dark image is loaded instead
        """
        self.default_threshold = 0.8
        self.default_dilate_iterations = 1
        self.default_gauss_ksize = 5
        self.bg_h, self.bg_w = 513, 513
        # Load background image, if path is None, use dark background
        self.bgd = load_bgd(bg_img_path, self.bg_w, self.bg_h,
                            post_process=lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # Initialize tflite-interpreter
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.input_shape = input_details[0]['shape'][1:3]
        self.input_node, self.output_node = input_details[0]['index'], output_details[0]['index']

    def load_new_bgd(self, bg_img_path):
        self.bgd = load_bgd(bg_img_path, self.bg_w, self.bg_h,
                            post_process=lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def segment_frame(self, frame, disp_wh):
        """
        segment person in frame and return frame as foreground and img from bg_img_path as background
        args:
            frame: numpy.ndarray, original numpy array image loaded with cv2 in BGR fmt
            disp_wh: tuple, final width and height of processed frame (disp_w, disp_h)
        """
        in_h, in_w = self.input_shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        simg = cv2.resize(img, (in_w, in_h),
                          interpolation=cv2.INTER_AREA) / 255.0
        simg = np.expand_dims(simg, axis=0).astype(np.float32)

        # predict Segmentation
        self.interpreter.set_tensor(self.input_node, simg)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_node)

        # 0: background channel, 1: foreground channel, 0 is more stable
        msk = out[0][:, :, 0]
        # since we are post-processing the bg mask
        msk = cv2.dilate(msk, np.ones((3, 3), dtype=np.uint8),
                         iterations=self.default_dilate_iterations)
        msk = cv2.GaussianBlur(msk,
                               ksize=(self.default_gauss_ksize,
                                      self.default_gauss_ksize),
                               sigmaX=4,
                               sigmaY=0)
        frame = get_frame_after_postprocess(
            msk, img, self.bgd, (self.bg_w, self.bg_h), disp_wh, self.default_threshold, foreground="bgd")
        return frame
