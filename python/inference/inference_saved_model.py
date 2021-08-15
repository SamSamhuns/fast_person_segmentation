# loading tensorflow v1 model
import cv2
import numpy as np
from time import time
from enum import Enum
from utils.inference import get_cmd_argparser, get_config_dict, load_bgd

import tensorflow as tf
tf.config.optimizer.set_jit(True)


class Post_Processing(Enum):
    """Post_Processing methods
    """
    GAUSSIAN = "gaussian"
    MORPH_OPEN = "morph_open"


def inference_model(vid_path,
                    bg_img_path,
                    pb_model_path,
                    json_config_path="models/model_info.json"):
    # choose parameters
    post_processing = Post_Processing.GAUSSIAN
    default_threshold = 0.63
    default_mopen_ksize = 7
    default_mopen_iter = 9
    default_gauss_ksize = 3
    bg_h, bg_w = 513, 513
    disp_h, disp_w = 720, 1200

    # load model config from json file
    config_dict = get_config_dict(pb_model_path, json_config_path)
    in_h, in_w = config_dict["in_height"], config_dict["in_width"]

    # Load background image, if path is None, use dark background
    def post_process(img): return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bgd = load_bgd(bg_img_path, bg_w, bg_h, post_process=post_process)

    # load video
    vid_path = 0 if vid_path is None else vid_path

    model = tf.saved_model.load(pb_model_path)
    # should be ["serving_default"]
    print("signature keys: ", list(model.signatures.keys()))
    model_inference = model.signatures["serving_default"]
    print("model output name & structure: ",
          model_inference.structured_outputs)

    cv2_disp_name = post_processing.name
    cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()
    fps = ""

    while ret:
        # Capture frame-by-frame
        t1 = time()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        simg = cv2.resize(img, (in_w, in_h),
                          interpolation=cv2.INTER_AREA) / 255.0
        # Predict
        simg = np.expand_dims(simg, axis=0)
        out = model_inference(tf.constant(simg, dtype=tf.float32))["op"]

        """ MORPH_OPEN SMOOTHING """
        if post_processing == Post_Processing.MORPH_OPEN:
            msk = np.float32(out).reshape((in_h, in_w, 1))
            msk = cv2.resize(msk, (bg_w, bg_h),
                             interpolation=cv2.INTER_LINEAR).reshape((bg_h, bg_w, 1))

            # default kernel size (10, 10) and iterations =10
            msk = cv2.morphologyEx(msk,
                                   cv2.MORPH_OPEN,
                                   ksize=(default_mopen_ksize,
                                          default_mopen_ksize),
                                   iterations=default_mopen_iter).reshape(
                (bg_h, bg_w, 1)) > default_threshold

        """ GAUSSIAN SMOOTHING """
        if post_processing == Post_Processing.GAUSSIAN:
            msk = np.float32(out).reshape((in_h, in_w, 1))
            msk = cv2.GaussianBlur(msk,
                                   ksize=(default_gauss_ksize,
                                          default_gauss_ksize),
                                   sigmaX=4,
                                   sigmaY=0)

            msk = cv2.resize(msk,
                             (bg_w, bg_h)).reshape((bg_h, bg_w, 1)) > default_threshold

        # Post-process
        img = cv2.resize(img, (bg_w, bg_h))

        # Alpha blending: (img * msk) + (bgd * (1 - msk))
        frame = np.where(msk, img, bgd).astype(np.uint8)

        # resize to final resolution
        frame = cv2.resize(frame, (disp_w, disp_h),
                           interpolation=cv2.INTER_LINEAR)

        # Display the resulting frame & FPS
        cv2.putText(frame, fps, (disp_h - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(cv2_disp_name, frame[..., ::-1])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()
        fps = f"FPS: {1/(time() - t1):.1f}"


def main():
    parser = get_cmd_argparser(
        default_model="models/deconv_bnoptimized_munet_e260")
    args = parser.parse_args()
    inference_model(args.source_vid_path,
                    args.bg_img_path,
                    args.model_path)


if __name__ == "__main__":
    main()
