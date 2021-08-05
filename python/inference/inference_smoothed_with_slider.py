from tensorflow.keras.models import load_model
from time import time
import numpy as np
import cv2

# custom util import
from utils.inference import get_cmd_argparser, get_config_dict, load_bgd


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

    default_threshold = 13
    default_skip_frame = 0
    default_mopen_ksize = 7
    default_mopen_iter = 9

    bg_h, bg_w = 360, 640             # background height & width
    disp_h, disp_w = 720, 1280        # final display screen width and height
    # Load background image, if path is None, use dark background
    bgd = load_bgd(bg_img_path, bg_w, bg_h)

    # cv2 video capture
    cap = cv2.VideoCapture(vid_path)

    fps = ""  # var to sotre FPS display text
    msk = None  # mask variable to store previous masked state
    COUNT = 0  # should always be set to 0

    cv2_disp_name = "new_morph_smooth"
    # Trackbars
    cv2.namedWindow(cv2_disp_name)
    # create trackbars for gaussian kernel
    cv2.createTrackbar('threshold', cv2_disp_name,
                       default_threshold, 20,
                       nothing)  # default 12/20 = 0.6
    cv2.createTrackbar('skip_frame', cv2_disp_name,
                       default_skip_frame, 5, nothing)  # default 2
    cv2.createTrackbar('mopen_ksize', cv2_disp_name,
                       default_mopen_ksize, 20, nothing)  # default 2
    cv2.createTrackbar('mopen_iter', cv2_disp_name,
                       default_mopen_iter, 20, nothing)  # default 2

    while True:
        # Capture frame-by-frame
        t1 = time()
        ret, frame = cap.read()

        # get current positions of trackbars
        p_thres = cv2.getTrackbarPos('threshold', cv2_disp_name) / 20
        skip_frame = cv2.getTrackbarPos('skip_frame', cv2_disp_name) + 1
        mopen_kernel = cv2.getTrackbarPos('mopen_ksize', cv2_disp_name)
        mopen_iter = cv2.getTrackbarPos('mopen_iter', cv2_disp_name)

        if ret:
            # Preprocess
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            simg = cv2.resize(img, (in_h, in_w), interpolation=cv2.INTER_AREA)
            simg = simg.reshape((1, in_h, in_w, 3)) / 255.0

            if skip_frame == 0 or COUNT % skip_frame == 0:
                # Predict
                out = model.predict(simg)
                msk = np.float32(out).reshape((in_h, in_w, 1))
                msk = cv2.resize(msk, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR).reshape(
                    (bg_h, bg_w, 1))

                msk = cv2.morphologyEx(msk,
                                       cv2.MORPH_OPEN,
                                       (mopen_kernel, mopen_kernel),
                                       iterations=mopen_iter).reshape((bg_h, bg_w, 1)) > p_thres
                COUNT = 0
            COUNT += 1

            # rescaling resize
            img = cv2.resize(img, (bg_w, bg_h)) / 255.0
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
            cv2.imshow(cv2_disp_name, frame[..., ::-1])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fps = f"FPS: {1/(time() - t1):.1f}"

    print("Final values upon exit")
    print(f"mopen_ksize={mopen_kernel}", f"mopen_iter={mopen_iter}",
          f"p_thres={p_thres}", f"skip_frame={skip_frame}")
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = get_cmd_argparser()
    args = parser.parse_args()
    inference_model(args.source_vid_path,
                    args.bg_img_path,
                    args.model_path)


if __name__ == "__main__":
    main()
