import cv2
import numpy as np
from time import time
from enum import Enum
import tensorflow as tf
from tensorflow.python.platform import gfile

# custom imports
from utils.inference import get_cmd_argparser, get_config_dict
from utils.inference import load_bgd, VideoStreamMultiThreadWidget
tf.config.optimizer.set_jit(True)


class Post_Processing(Enum):
    """Post_Processing methods
    """
    GAUSSIAN = "gaussian"
    MORPH_OPEN = "morph_open"


def inference_model(vid_path,
                    bg_img_path,
                    pb_model_path,
                    multi_thread=True,
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

    sess = tf.compat.v1.Session()
    # load pb model and params
    f = gfile.FastGFile(pb_model_path, 'rb')
    input_node = config_dict["in_name"]
    output_layer = config_dict["out_name"]

    graph_def = tf.compat.v1.GraphDef()
    # Parses a serialized binary message into the current message.
    graph_def.ParseFromString(f.read())
    f.close()

    sess.graph.as_default()
    # Import a serialized TensorFlow `GraphDef` protocol buffer
    # and place into the current default `Graph`.
    tf.import_graph_def(graph_def)

    # print graph nodes if required
    # graph_nodes = [n.name for n in graph_def.node]
    # print(graph_nodes)

    with tf.compat.v1.Session() as sess:
        prob_tensor = sess.graph.get_tensor_by_name(output_layer)
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
            # Predict
            out = sess.run(prob_tensor, {input_node: [simg]})

            msk = np.float32(out).reshape((in_h, in_w, 1))
            """ MORPH_OPEN SMOOTHING """
            if post_processing == Post_Processing.MORPH_OPEN:
                kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(default_mopen_ksize, default_mopen_ksize))
                msk = cv2.morphologyEx(msk,
                                       cv2.MORPH_OPEN,
                                       kernel=kernel,
                                       iterations=default_mopen_iter)

            """ GAUSSIAN SMOOTHING """
            if post_processing == Post_Processing.GAUSSIAN:
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
            frame = np.where(msk, img, bgd).astype(np.uint8)

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
        default_model="models/transpose_seg/deconv_bnoptimized_munet_e260.pb")
    args = parser.parse_args()
    inference_model(vid_path=args.source_vid_path,
                    bg_img_path=args.bg_img_path,
                    pb_model_path=args.model_path,
                    multi_thread=args.use_multi_thread)


if __name__ == "__main__":
    main()
