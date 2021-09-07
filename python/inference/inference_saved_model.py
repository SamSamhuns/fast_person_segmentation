import cv2
import numpy as np
from time import time

from utils.inference import load_bgd, get_cmd_argparser, get_config_dict, get_frame_after_postprocess
from utils.inference import PostProcessingType, ImageioVideoWriter, get_video_stream_widget


import tensorflow as tf
tf.config.optimizer.set_jit(True)


def inference_model(vid_path,
                    bg_img_path,
                    pb_model_path,
                    json_config_path="models/model_info.json",
                    multi_thread=True,
                    output_dir=None):
    # choose parameters
    post_processing = PostProcessingType.GAUSSIAN
    default_threshold = 0.8
    default_mopen_ksize = 7
    default_mopen_iter = 9
    default_gauss_ksize = 3
    bg_h, bg_w = 513, 513
    disp_h, disp_w = 720, 1200

    # load model config from json file
    # config_dict = get_config_dict(pb_model_path, json_config_path)
    # in_h, in_w = config_dict["in_height"], config_dict["in_width"]
    in_h, in_w = 144, 256

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
    # replace the output node name
    output_node_name = "segment"

    cv2_disp_name = post_processing.name
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
        # Capture frame-by-frame
        t1 = time()
        # for handling multi_threading load
        ret, frame = cap.read()
        if frame is None:
            continue
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        simg = cv2.resize(img, (in_w, in_h),
                          interpolation=cv2.INTER_AREA) / 255.0
        # Predict
        simg = np.expand_dims(simg, axis=0)
        out = model_inference(tf.constant(simg, dtype=tf.float32))[output_node_name]

        # 0: background channel, 1: foreground channel, 0 is more stable
        msk = out[0][:, :, 0]
        msk = np.float32(msk)
        """ MORPH_OPEN SMOOTHING """
        if post_processing == PostProcessingType.MORPH_OPEN:
            kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(default_mopen_ksize, default_mopen_ksize))
            msk = cv2.morphologyEx(msk,
                                   cv2.MORPH_OPEN,
                                   kernel=kernel,
                                   iterations=default_mopen_iter)

        """ GAUSSIAN SMOOTHING """
        if post_processing == PostProcessingType.GAUSSIAN:
            msk = cv2.GaussianBlur(msk,
                                   ksize=(default_gauss_ksize,
                                          default_gauss_ksize),
                                   sigmaX=4,
                                   sigmaY=0)
        # postprocess
        frame = get_frame_after_postprocess(
            msk, img, bgd, (bg_w, bg_h), (disp_w, disp_h), default_threshold, foreground="bgd")

        # Display the resulting frame & FPS
        cv2.putText(frame, fps, (disp_h - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(cv2_disp_name, frame[..., ::-1])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        vwriter.write_frame(frame) if output_dir else None
        fps = f"FPS: {1/(time() - t1):.1f}"
    vwriter.close() if output_dir else None
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = get_cmd_argparser(
        default_model="models/selfie_seg/144x256")
    args = parser.parse_args()
    inference_model(args.source_vid_path,
                    args.bg_img_path,
                    args.model_path,
                    multi_thread=args.use_multi_thread,
                    output_dir=args.output_dir)


if __name__ == "__main__":
    main()
