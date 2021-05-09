# loading tensorflow v1 model
import cv2
import numpy as np
from time import time
from enum import Enum
import tensorflow as tf
from tensorflow.python.platform import gfile

# custom imports
from utils.inference import get_parsed_cmd_args, get_config_dict, load_bgd


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
    disp_h, disp_w = 1200, 720

    # load model config from json file
    config_dict = get_config_dict(pb_model_path, json_config_path)
    in_h, in_w = config_dict["in_height"], config_dict["in_width"]

    # Load background image, if path is None, use dark background
    bgd = load_bgd(bg_img_path, bg_w, bg_h)

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
    # graph_nodes  = [n.name for n in graph_def.node]
    # print(graph_nodes)

    with tf.compat.v1.Session() as sess:
        try:
            prob_tensor = sess.graph.get_tensor_by_name(output_layer)
            cv2_disp_name = post_processing.name

            cap = cv2.VideoCapture(vid_path)
            fps = ""

            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                t1 = time()

                if ret:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    simg = cv2.resize(img, (in_h, in_w),
                                      interpolation=cv2.INTER_AREA) / 255.0
                    # Predict
                    out = sess.run(prob_tensor, {input_node: [simg]})

                    """ MORPH_OPEN SMOOTHING """
                    if post_processing == Post_Processing.MORPH_OPEN:
                        msk = np.float32(out).reshape((in_h, in_w, 1))
                        msk = cv2.resize(msk, (bg_h, bg_w),
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
                                         (bg_h, bg_w)).reshape((bg_h, bg_w, 1)) > default_threshold

                    # Post-process
                    img = cv2.resize(img, (bg_h, bg_w)) / 255.0

                    # Alpha blending
                    frame = (img * msk) + (bgd * (1 - msk))

                    # resize to final resolution
                    frame = np.uint8(frame * 255.0)
                    frame = cv2.resize(frame, (disp_h, disp_w),
                                       interpolation=cv2.INTER_LINEAR)

                    # Display the resulting frame & FPS
                    cv2.putText(frame, fps, (disp_h - 180, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow(cv2_disp_name, frame[..., ::-1])
                    fps = f"FPS: {1/(time() - t1):.1f}"

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        except Exception as e:
            print(e)


def main():
    args = get_parsed_cmd_args(
        default_model="models/transpose_seg/deconv_bnoptimized_munet_e260.pb")
    inference_model(args.source_vid_path,
                    args.bg_img_path,
                    args.model_path)


if __name__ == "__main__":
    main()
