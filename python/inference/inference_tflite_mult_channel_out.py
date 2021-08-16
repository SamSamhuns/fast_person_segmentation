import cv2
import numpy as np
import tensorflow as tf

from time import time
from enum import Enum
from utils.inference import get_cmd_argparser, load_bgd
from utils.inference import VideoStreamMultiThreadWidget


class Post_Processing(Enum):
    """Post_Processing methods
    """
    GAUSSIAN = "gaussian"
    MORPH_OPEN = "morph_open"


def inference_model(vid_path,
                    bg_img_path,
                    tflite_model_path,
                    multi_thread=True):
    # choose parameters
    post_processing = Post_Processing.GAUSSIAN
    default_threshold = 0.8
    default_mopen_ksize = 7
    default_mopen_iter = 9
    default_gauss_ksize = 5
    bg_h, bg_w = 513, 513
    disp_h, disp_w = 720, 1200

    # Load background image, if path is None, use dark background
    def post_process(img): return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    print(f"Model input shape {in_h, in_w}")
    cv2_disp_name = post_processing.name
    # check if multi-threading is to be used
    if multi_thread:
        cap = VideoStreamMultiThreadWidget(vid_path)
    else:
        cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()
    fps = ""
    while ret:
        t1 = time()
        # for handling multi_threading load
        try:
            ret, frame = cap.read()
            if frame is None:
                raise AttributeError
        except AttributeError:
            continue
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        simg = cv2.resize(img, (in_w, in_h),
                          interpolation=cv2.INTER_AREA) / 255.0
        simg = np.expand_dims(simg, axis=0).astype(np.float32)

        # Predict Segmentation
        interpreter.set_tensor(input_node, simg)
        interpreter.invoke()
        out = interpreter.get_tensor(output_node)

        # 0: background channel, 1: foreground channel, 0 is more stable
        msk = out[0][:, :, 0]

        """ MORPH_OPEN SMOOTHING """
        if post_processing == Post_Processing.MORPH_OPEN:
            # since we are post-processing the bg mask
            msk = cv2.dilate(msk, np.ones((3, 3), dtype=np.uint8), iterations=1)
            kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(default_mopen_ksize, default_mopen_ksize))
            msk = cv2.morphologyEx(msk,
                                   cv2.MORPH_OPEN,
                                   kernel=kernel,
                                   iterations=default_mopen_iter)

        """ GAUSSIAN SMOOTHING """
        if post_processing == Post_Processing.GAUSSIAN:
            # since we are post-processing the bg mask
            msk = cv2.dilate(msk, np.ones((3, 3), dtype=np.uint8), iterations=1)
            msk = cv2.GaussianBlur(msk,
                                   ksize=(default_gauss_ksize,
                                          default_gauss_ksize),
                                   sigmaX=4,
                                   sigmaY=0)
        msk = cv2.resize(
            msk, (bg_w, bg_h)).reshape((
                bg_h, bg_w, 1)) > default_threshold

        # Post-process
        img = cv2.resize(img, (bg_w, bg_h))

        # Alpha blending: (img * msk) + (bgd * (1 - msk))
        frame = np.where(msk, bgd, img).astype(np.uint8)

        # resize to final resolution
        frame = cv2.resize(frame, (disp_w, disp_h),
                           interpolation=cv2.INTER_LINEAR)

        # Display the resulting frame & FPS
        cv2.putText(frame, fps, (disp_h - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(cv2_disp_name, frame[..., ::-1])

        if cv2.waitKey(1) == 113:  # press "q" to stop
            break
        fps = f"FPS: {1/(time() - t1):.1f}"
    cv2.destroyAllWindows()


def main():
    parser = get_cmd_argparser(
        default_model="models/transpose_seg/deconv_fin_munet.tflite")
    args = parser.parse_args()
    inference_model(vid_path=args.source_vid_path,
                    bg_img_path=args.bg_img_path,
                    tflite_model_path=args.model_path,
                    multi_thread=args.use_multi_thread)


if __name__ == "__main__":
    main()