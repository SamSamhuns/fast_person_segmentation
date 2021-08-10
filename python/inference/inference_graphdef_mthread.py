import cv2
import numpy as np
from enum import Enum
import tensorflow as tf
from time import time, sleep
from threading import Thread
from tensorflow.python.platform import gfile

# custom imports
from utils.inference import get_cmd_argparser, get_config_dict, load_bgd
tf.config.optimizer.set_jit(True)


class Post_Processing(Enum):
    """Post_Processing methods
    """
    GAUSSIAN = "gaussian"
    MORPH_OPEN = "morph_open"


class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            sleep(.01)

    def show_frame(self):
        # Display frames in main program
        cv2.imshow('frame', self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)


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

        video_stream_widget = VideoStreamWidget(vid_path)
        fps = ""
        while video_stream_widget.capture.isOpened():
            # Capture frame-by-frame
            t1 = time()
            try:
                frame = video_stream_widget.frame
            except AttributeError:
                continue
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
            img = cv2.resize(img, (bg_h, bg_w))

            # Alpha blending: (img * msk) + (bgd * (1 - msk))
            frame = np.where(msk, img, bgd).astype(np.uint8)

            # resize to final resolution
            frame = cv2.resize(frame, (disp_h, disp_w),
                               interpolation=cv2.INTER_LINEAR)

            # Display the resulting frame & FPS
            cv2.putText(frame, fps, (disp_h - 180, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(cv2_disp_name, frame[..., ::-1])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fps = f"FPS: {1/(time() - t1):.1f}"


def main():
    parser = get_cmd_argparser(
        default_model="models/transpose_seg/deconv_bnoptimized_munet_e260.pb")
    args = parser.parse_args()
    inference_model(args.source_vid_path,
                    args.bg_img_path,
                    args.model_path)


if __name__ == "__main__":
    main()
