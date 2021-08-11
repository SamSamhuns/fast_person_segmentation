import cv2
import numpy as np
import mediapipe as mp

from time import time
from enum import Enum
from utils.inference import get_cmd_argparser, load_bgd
from utils.inference import VideoStreamMultiThreadWidget


class InferenceMode(Enum):
    """Inference Mode
    """
    IMAGE = "image"
    VIDEO = "video"


class BackgroundMode(Enum):
    """bBackground Mode
    """
    BLUR = "blur"


mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_holistic = mp.solutions.holistic
mp_objectron = mp.solutions.objectron

thres = 0.34


def image_inference(image_path_list):
    MASK_COLOR = (255, 255, 255)  # white
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:

        for idx, file in enumerate(image_path_list):
            image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = selfie_segmentation.process(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Draw selfie segmentation on the background image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            condition = np.stack(
                (results.segmentation_mask,) * 3, axis=-1) > thres
            # Generate solid color images for showing the output selfie segmentation mask.
            fg_image = np.zeros(image.shape, dtype=np.uint8)
            fg_image[:] = MASK_COLOR
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            output_image = np.where(condition, fg_image, bg_image)
            cv2.imwrite('/tmp/selfie_segmentation_output' +
                        str(idx) + '.png', output_image)


def video_inference(video_src, bg_image_path, bg_mode=None, multi_thread=False):
    """
    video_src: path to video file or use 0 for webcam
    """
    video_src = 0 if video_src is None else video_src
    disp_h, disp_w = 720, 1280
    bg_image = load_bgd(bg_image_path, disp_w, disp_h,
                        dtype=np.uint8, post_process=None)

    # check if multi-threading is to be used
    if multi_thread:
        cap = VideoStreamMultiThreadWidget(video_src)
    else:
        cap = cv2.VideoCapture(video_src)
    ret, frame = cap.read()
    fps = ""
    # model_selection=1 uses landscape mode
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        while ret:
            itime = time()
            # for handling multi_threading load
            try:
                ret, frame = cap.read()
                if frame is None:
                    raise AttributeError
            except AttributeError:
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = selfie_segmentation.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw selfie segmentation on the background image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            mask = results.segmentation_mask
            # mask = np.expand_dims(results.segmentation_mask, -1)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (17, 17), iterations=9)
            # mask = cv2.GaussianBlur(mask, ksize=(3, 3), sigmaX=4, sigmaY=0)
            # mask = cv2.ximgproc.jointBilateralFilter(mask, image, 3, 5, 5)

            condition = np.stack((mask,) * 3, axis=-1) > thres
            # The background can be customized.
            #   a) Load an image (with the same width and height of the input image) to
            #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
            #   b) Blur the input image by applying image filtering, e.g.,
            #      bg_image = cv2.GaussianBlur(image,(55,55),0)
            if bg_mode is not None:
                if bg_mode == BackgroundMode.BLUR:
                    bg_image = cv2.GaussianBlur(image, (65, 65), 0)
            output_image = np.where(condition, image, bg_image)

            # Display the resulting frame & FPS
            cv2.putText(output_image, fps, (disp_h - 180, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Selfie Segmentation', output_image)
            if cv2.waitKey(1) == 113:  # press "q" to stop
                break
            fps = f"FPS: {1/(time() - itime):.1f}"
    cv2.destroyAllWindows()


def inference_model(mode, **kwargs):
    if mode == InferenceMode.IMAGE:
        image_inference(**kwargs)
    elif mode == InferenceMode.VIDEO:
        video_inference(**kwargs, bg_mode=None)


def main():
    parser = get_cmd_argparser(default_model=None)
    args = parser.parse_args()
    inference_model(mode=InferenceMode.VIDEO,
                    video_src=args.source_vid_path,
                    bg_image_path=args.bg_img_path,
                    multi_thread=args.use_multi_thread)


if __name__ == "__main__":
    main()
