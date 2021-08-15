from keras.models import load_model
import numpy as np
import cv2

# custom util import
from utils.inference import get_cmd_argparser, get_config_dict, load_bgd


def inference_model(vid_path,
                    bg_img_path,
                    model_path,
                    json_config_path="models/model_info.json"):
    # load model config from json file
    config_dict = get_config_dict(model_path, json_config_path)
    input_height, input_width = config_dict["in_height"], config_dict["in_width"]

    # load model
    model = load_model(model_path, compile=False)
    # load video
    vid_path = 0 if vid_path is None else vid_path

    # hyper-parameters
    ksize = 3                   # gaussian smoothing kernel size
    p_thres = 0.75              # threshold for prediction
    bg_h, bg_w = 513, 513       # background height & width
    disp_h, disp_w = 640, 1000  # final display screen width and height
    # Load background image, if path is None, use dark background
    bgd = load_bgd(bg_img_path, bg_w, bg_h)

    cap = cv2.VideoCapture(vid_path)
    msk = None  # mask variable to store previous masked state
    ret, frame = cap.read()
    while ret:
        # Preprocess
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        simg = cv2.resize(img, (input_width, input_height),
                          interpolation=cv2.INTER_AREA)
        simg = simg.reshape((1, input_height, input_width, 3)) / 255.0

        # Predict
        out = model.predict(simg)
        # Post-process
        msk = np.float32((out > p_thres)).reshape(
            (input_height, input_width, 1))
        msk = cv2.GaussianBlur(msk, ksize=(
            ksize, ksize), sigmaX=4, sigmaY=0)

        msk = cv2.resize(msk, (bg_w, bg_h)).reshape((bg_h, bg_w, 1))
        img = cv2.resize(img, (bg_w, bg_h)) / 255.0

        # Alpha blending
        frame = (img * msk) + (bgd * (1 - msk))

        # resize to final resolution
        frame = np.uint8(frame * 255.0)
        frame = cv2.resize(frame, (disp_w, disp_h),
                           interpolation=cv2.INTER_LINEAR)

        # Display the resulting frame
        cv2.imshow('image', frame[..., ::-1])
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()

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
