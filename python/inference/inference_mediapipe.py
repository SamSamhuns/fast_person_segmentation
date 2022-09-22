from time import time
from typing import Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp

from utils.inference import get_cmd_argparser, load_bgd, get_video_stream_widget
from utils.inference import ImageioVideoWriter, PostProcessingType, BackgroundMode


mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_holistic = mp.solutions.holistic
mp_objectron = mp.solutions.objectron


def inference_video(vid_path: str,
                    bg_image_path: str,
                    bg_mode: Optional[BackgroundMode] = None,
                    disp_wh_size: Tuple[int, int] = (1280, 720),
                    multi_thread: bool = False,
                    output_dir: Optional[str] = None):
    """
    vid_path: path to video file or use 0 for webcam
    """
    post_processing = PostProcessingType.GAUSSIAN
    default_threshold = 0.8
    default_mopen_ksize = 7
    default_mopen_iter = 9
    default_gauss_ksize = 17
    cv2_disp_name = "mediapipe_" + post_processing.name

    vid_path = 0 if vid_path is None else vid_path
    disp_w, disp_h = disp_wh_size

    # check if multi-threading is to be used
    if multi_thread:
        cap = get_video_stream_widget(vid_path)
    else:
        cap = cv2.VideoCapture(vid_path)
    if output_dir is not None:
        vwriter = ImageioVideoWriter(output_dir, str(vid_path), __file__)

    ret, frame = cap.read()
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bg_image = load_bgd(bg_image_path, frame_w, frame_h,
                        dtype=np.uint8, post_process=None)
    fps = ""
    # model_selection=1 uses landscape mode
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        while ret:
            itime = time()
            ret, frame = cap.read()
            if frame is None:
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

            # Mask PostProcessing
            if post_processing == PostProcessingType.MORPH_OPEN:
                kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(
                    default_mopen_ksize, default_mopen_ksize))
                mask = cv2.morphologyEx(
                    mask,
                    cv2.MORPH_OPEN,
                    kernel=kernel,
                    iterations=default_mopen_iter)
            elif post_processing == PostProcessingType.GAUSSIAN:
                mask = cv2.GaussianBlur(
                    mask,
                    ksize=(default_gauss_ksize,
                           default_gauss_ksize),
                    sigmaX=4,
                    sigmaY=0)

            condition = np.stack((mask,) * 3, axis=-1) > default_threshold
            # modify bg image if bg_mode set
            if bg_mode is not None:
                if bg_mode == BackgroundMode.BLUR:
                    bg_image = cv2.GaussianBlur(image, (65, 65), 0)

            output_image = np.where(condition, image, bg_image)
            output_image = cv2.resize(output_image, (disp_w, disp_h))

            # Display the resulting frame & FPS
            cv2.putText(output_image, fps, (disp_h - 180, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(cv2_disp_name, output_image)
            if cv2.waitKey(1) == 113:  # press "q" to stop
                break
            vwriter.write_frame(
                output_image[..., ::-1]) if output_dir else None
            fps = f"FPS: {1/(time() - itime):.1f}"
        vwriter.close() if output_dir else None
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = get_cmd_argparser(default_model=None)
    args = parser.parse_args()
    args.disp_wh_size = tuple(map(int, args.disp_wh_size))
    inference_video(vid_path=args.source_vid_path,
                    bg_image_path=args.bg_img_path,
                    disp_wh_size=args.disp_wh_size,
                    multi_thread=args.use_multi_thread,
                    output_dir=args.output_dir)


if __name__ == "__main__":
    main()
