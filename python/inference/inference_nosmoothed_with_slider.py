from keras.models import load_model
from time import time
import numpy as np
import cv2

# custom util import
from utils.inference import get_parsed_cmd_args, get_config_dict, load_bgd


def nothing(x):
    pass


def inference_model(vid_path,
                    bg_img_path,
                    model_path,
                    json_config_path="models/model_info.json"):
    # load model config from json file
    config_dict = get_config_dict(model_path, json_config_path)
    in_h, in_w = config_dict["in_height"], config_dict["in_width"]

    # load model
    model = load_model(model_path, compile=False)
    # load video
    vid_path = 0 if vid_path is None else vid_path

    default_ksize = 3
    default_threshold = 15
    default_skip_frame = 0

    bg_h, bg_w = 513, 513             # background height & width
    disp_h, disp_w = 720, 1280        # final display screen width and height

    # Load background image, if path is None, use dark background
    bgd = load_bgd(bg_img_path, bg_w, bg_h)

    # cv2 vidoe capture
    cap = cv2.VideoCapture(vid_path)

    fps = ""  # var to sotre FPS display text
    COUNT = 0  # should always be set to 0
    msk = None  # mask variable to store previous masked state

    cv2_disp_name = "orig_gaussian"
    # Trackbars
    cv2.namedWindow(cv2_disp_name)

    # create trackbars for gaussian kernel
    cv2.createTrackbar('ksize', cv2_disp_name, default_ksize,
                       30, nothing)      # default 4
    cv2.createTrackbar('threshold', cv2_disp_name, default_threshold, 20,
                       nothing)  # default 12/20 = 0.6
    cv2.createTrackbar('skip_frame', cv2_disp_name,
                       default_skip_frame, 5, nothing)  # default 2

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        t1 = time()

        # get current positions of trackbars
        p_thres = cv2.getTrackbarPos('threshold', cv2_disp_name) / 20
        skip_frame = cv2.getTrackbarPos('skip_frame', cv2_disp_name) + 1
        ksize = cv2.getTrackbarPos('ksize', cv2_disp_name)
        ksize = ksize if ksize & 1 else ksize + 1  # ksize must always be odd

        if ret:
            # Preprocess
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            simg = cv2.resize(img, (in_h, in_w), interpolation=cv2.INTER_AREA)
            simg = simg.reshape((1, in_h, in_w, 3)) / 255.0

            if skip_frame == 0 or COUNT % skip_frame == 0:
                # Predict
                out = model.predict(simg)
                msk = np.float32((out > p_thres)).reshape((in_h, in_w, 1))
                msk = cv2.GaussianBlur(msk,
                                       ksize=(ksize, ksize),
                                       sigmaX=4,
                                       sigmaY=0)
                msk = cv2.resize(msk, (bg_h, bg_w)).reshape((bg_h, bg_w, 1))
                COUNT = 0
            COUNT += 1

            # rescaling resize
            img = cv2.resize(img, (bg_h, bg_w)) / 255.0

            # Alpha blending
            frame = (img * msk) + (bgd * (1 - msk))

            # resize to final resolution
            frame = np.uint8(frame * 255.0)
            frame = cv2.resize(frame, (disp_w, disp_h),
                               interpolation=cv2.INTER_LINEAR)

            # Display the resulting frame & FPS
            cv2.putText(frame, fps, (disp_h - 180, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Thres=' + str(p_thres), (disp_h - 180, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('orig_gaussian', frame[..., ::-1])
            fps = f"FPS: {1/(time() - t1):.1f}"

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

    print("Final values upon exit")
    print(f"ksize={ksize}", f"p_thres={p_thres}",
          f"skip_frame={skip_frame}")
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_parsed_cmd_args()
    inference_model(args.source_vid_path,
                    args.bg_img_path,
                    args.model_path)


if __name__ == "__main__":
    main()
