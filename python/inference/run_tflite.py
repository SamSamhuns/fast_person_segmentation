from time import time
from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

from utils.inference import load_bgd, get_cmd_argparser, get_frame_after_postprocess
from utils.inference import PostProcessingType, ImageioVideoWriter, get_video_stream_widget


def inference_video(vid_path: str,
                    bg_img_path: str,
                    tflite_model_path: str,
                    disp_wh_size: Tuple[int, int] = (1280, 720),
                    multi_thread: bool = True,
                    output_dir: Optional[str] = None):
    # choose parameters
    post_processing = PostProcessingType.GAUSSIAN
    default_threshold = 0.63
    default_mopen_ksize = 7
    default_mopen_iter = 9
    default_gauss_ksize = 3
    bg_w, bg_h = 513, 513
    disp_w, disp_h = disp_wh_size
    cv2_disp_name = "tflite_" + post_processing.name

    # Load background image, if path is None, use dark background
    def post_process(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bgd = load_bgd(bg_img_path, bg_w, bg_h, post_process=post_process)

    # load video
    vid_path = 0 if vid_path is None else vid_path

    # Initialize tflite-interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape'][1:3]
    input_node, output_node = input_details[0]['index'], output_details[0]['index']

    in_h, in_w = input_shape[0], input_shape[1]
    # check if multi-threading is to be used
    if multi_thread:
        cap = get_video_stream_widget(vid_path)
    else:
        cap = cv2.VideoCapture(vid_path)
    if output_dir is not None:
        vwriter = ImageioVideoWriter(output_dir, str(vid_path), __file__)

    ret, frame = cap.read()
    fps = ""
    while ret:
        t1 = time()
        # for handling multi_threading load
        ret, frame = cap.read()
        if frame is None:
            continue

        # preprocess
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        simg = cv2.resize(img, (in_w, in_h),
                          interpolation=cv2.INTER_AREA) / 255.0
        simg = np.expand_dims(simg, axis=0).astype(np.float32)

        # predict segmentation
        interpreter.set_tensor(input_node, simg)
        interpreter.invoke()
        out = interpreter.get_tensor(output_node)

        msk = np.float32(out).reshape((in_h, in_w, 1))

        # Mask PostProcessing
        if post_processing == PostProcessingType.MORPH_OPEN:
            kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(
                default_mopen_ksize, default_mopen_ksize))
            msk = cv2.morphologyEx(msk,
                                   cv2.MORPH_OPEN,
                                   kernel=kernel,
                                   iterations=default_mopen_iter)
        elif post_processing == PostProcessingType.GAUSSIAN:
            msk = cv2.GaussianBlur(msk,
                                   ksize=(default_gauss_ksize,
                                          default_gauss_ksize),
                                   sigmaX=4,
                                   sigmaY=0)
        # postprocess
        frame = get_frame_after_postprocess(
            msk, img, bgd, (bg_w, bg_h), (disp_w, disp_h), default_threshold)

        # Display the resulting frame & FPS
        cv2.putText(frame, fps, (disp_h - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(cv2_disp_name, frame[..., ::-1])

        if cv2.waitKey(1) == 113:  # press "q" to stop
            break
        vwriter.write_frame(frame) if output_dir else None
        fps = f"FPS: {1/(time() - t1):.1f}"
    vwriter.close() if output_dir else None
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = get_cmd_argparser(
        default_model="models/transpose_seg/deconv_fin_munet.tflite")
    args = parser.parse_args()
    args.disp_wh_size = tuple(map(int, args.disp_wh_size))
    inference_video(vid_path=args.source_vid_path,
                    bg_img_path=args.bg_img_path,
                    tflite_model_path=args.model_path,
                    disp_wh_size=args.disp_wh_size,
                    multi_thread=args.use_multi_thread,
                    output_dir=args.output_dir)


if __name__ == "__main__":
    main()
