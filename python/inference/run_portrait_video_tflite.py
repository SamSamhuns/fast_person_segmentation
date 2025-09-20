from time import time
from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

from utils.inference import get_cmd_argparser, remove_argparse_option
from utils.inference import ImageioVideoWriter, get_video_stream_widget


def normalize(
    img: np.array,
    scale: float = 1,
    mean=[103.94, 116.78, 123.68],
    val=[0.017, 0.017, 0.017],
) -> np.ndarray:
    img = img / scale
    return (img - mean) * val


def blend(frame: np.array, alpha: np.array):
    """Alpha blend frame with background"""
    background = np.zeros(frame.shape) + [0, 0, 0]
    alphargb = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    result = np.uint8(frame * alphargb + background * (1 - alphargb))
    return frame, alphargb * 255, result


def inference_video(
    vid_path: str,
    tflite_model_path: str,
    disp_wh_size: Tuple[int, int] = (1280, 720),
    multi_thread: bool = True,
    output_dir: Optional[str] = None,
):
    disp_w, disp_h = disp_wh_size
    cv2_disp_name = "Portrait Video tflite"

    # Initialize tflite-interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    in_h, in_w = input_details[0]["shape"][1:3]

    # load video
    vid_path = 0 if vid_path is None else vid_path

    # check if multi-threading is to be used
    if multi_thread:
        cap = get_video_stream_widget(vid_path)
    else:
        cap = cv2.VideoCapture(vid_path)

    if output_dir is not None:
        vwriter = ImageioVideoWriter(output_dir, str(vid_path), __file__)

    pred_video = None
    fps = ""
    count = 0

    ret, frame = cap.read()
    frame_h, frame_w = frame.shape[:2]

    while ret:
        t1 = time()
        image = cv2.resize(frame, (in_w, in_h))
        image = normalize(image)

        # Choose prior mask
        if count == 0:
            prior = np.zeros((in_h, in_w, 1))  # first frame
        else:
            prior = pred_video

        # Add prior as fourth channel
        image = np.dstack([image, prior])
        prepimg = image[np.newaxis, :, :, :]

        if count % 2 == 0:
            # Invoke interpreter for inference
            interpreter.set_tensor(
                input_details[0]["index"], np.array(prepimg, dtype=np.float32)
            )
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]["index"])
            output = output.reshape(in_h, in_w, 1)
        else:
            output = pred_video

        # Save output to feed subsequent inputs
        pred_video = output

        # Process the output
        output = cv2.resize(output, (frame_w, frame_h))
        _, _, output = blend(frame, output)

        # Display the resulting frame & FPS
        cv2.putText(
            output,
            fps,
            (frame_h - 180, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        output = cv2.resize(output, (disp_w, disp_h))
        cv2.imshow(cv2_disp_name, output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        vwriter.write_frame(output) if output_dir else None
        ret, frame = cap.read()
        fps = f"FPS: {1 / (time() - t1):.1f}"
    vwriter.close() if output_dir else None
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = get_cmd_argparser(
        default_model="models/segvid_mnv2_port256/portrait_video.tflite"
    )
    remove_argparse_option(parser, "bg_img_path")
    args = parser.parse_args()
    args.disp_wh_size = tuple(map(int, args.disp_wh_size))
    inference_video(
        vid_path=args.source_vid_path,
        tflite_model_path=args.model_path,
        disp_wh_size=args.disp_wh_size,
        multi_thread=args.use_multi_thread,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
