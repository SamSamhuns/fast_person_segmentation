from tensorflow.keras.models import load_model
from time import time
import numpy as np
import cv2

from utils.inference import load_bgd, get_cmd_argparser, get_config_dict, get_frame_after_postprocess
from utils.inference import ImageioVideoWriter, VideoStreamMultiThreadWidget


def nothing(x):
    pass


def inference_model(vid_path,
                    bg_img_path,
                    model_path,
                    multi_thread=True,
                    json_config_path="models/model_info.json",
                    output_dir=None):
    # choose parameters
    default_threshold = 13
    default_skip_frame = 0
    default_mopen_ksize = 7
    default_mopen_iter = 2
    bg_h, bg_w = 360, 640             # background height & width
    disp_h, disp_w = 720, 1280        # final display screen width and height

    # load model config from json file
    config_dict = get_config_dict(model_path, json_config_path)
    in_h, in_w = config_dict["in_height"], config_dict["in_width"]

    model = load_model(model_path, compile=False)  # load model
    vid_path = 0 if vid_path is None else vid_path  # load video

    # Load background image, if path is None, use dark background
    bgd = load_bgd(bg_img_path, bg_w, bg_h)

    fps = ""    # var to sotre FPS display text
    msk = None  # mask variable to store previous masked state
    COUNT = 0   # should always be set to 0

    cv2_disp_name = "new_morph_smooth"
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

    # check if multi-threading is to be used
    if multi_thread:
        cap = VideoStreamMultiThreadWidget(vid_path)
    else:
        cap = cv2.VideoCapture(vid_path)
    if output_dir is not None:
        vwriter = ImageioVideoWriter(output_dir, str(vid_path), __file__)
    ret, frame = cap.read()
    while ret:
        t1 = time()
        # for handling multi_threading load
        ret, frame = cap.read()
        if frame is None:
            continue

        # get current positions of trackbars
        threshold = cv2.getTrackbarPos('threshold', cv2_disp_name) / 20
        skip_frame = cv2.getTrackbarPos('skip_frame', cv2_disp_name) + 1
        mopen_kernel = cv2.getTrackbarPos('mopen_ksize', cv2_disp_name)
        mopen_iter = cv2.getTrackbarPos('mopen_iter', cv2_disp_name)

        # preprocess
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        simg = cv2.resize(img, (in_w, in_h),
                          interpolation=cv2.INTER_AREA) / 255.0
        simg = np.expand_dims(simg, axis=0).astype(np.float32)

        if skip_frame == 0 or COUNT % skip_frame == 0:
            # predict segmentation
            out = model.predict(simg)
            msk = np.float32(out).reshape((in_h, in_w, 1))
            msk = cv2.morphologyEx(msk,
                                   cv2.MORPH_OPEN,
                                   (mopen_kernel, mopen_kernel),
                                   iterations=mopen_iter)
            COUNT = 0
        COUNT += 1

        # postprocess
        frame = get_frame_after_postprocess(
            msk, img, bgd, (bg_w, bg_h), (disp_w, disp_h), threshold)

        # Display the resulting frame & FPS
        cv2.putText(frame, fps, (disp_h - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(cv2_disp_name, frame[..., ::-1])

        if cv2.waitKey(1) == 113:  # press "q" to stop
            break
        vwriter.write_frame(frame) if output_dir else None
        fps = f"FPS: {1 / (time() - t1):.1f}"

    print("Final values upon exit")
    print(f"mopen_ksize={mopen_kernel}", f"mopen_iter={mopen_iter}",
          f"threshold={threshold}", f"skip_frame={skip_frame}")
    vwriter.close() if output_dir else None
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = get_cmd_argparser()
    args = parser.parse_args()
    inference_model(vid_path=args.source_vid_path,
                    bg_img_path=args.bg_img_path,
                    model_path=args.model_path,
                    multi_thread=args.use_multi_thread,
                    output_dir=args.output_dir)


if __name__ == "__main__":
    main()
