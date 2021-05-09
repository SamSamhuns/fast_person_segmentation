import cv2
import time
import numpy as np
from PIL import Image
import tensorflow as tf

# custom util import
from utils.inference import get_parsed_cmd_args


def inference_model(vid_path,
                    model_path):
    # Initialize tflite-interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape'][1:3]

    # load video
    vid_path = 0 if vid_path is None else vid_path

    # Width and height
    width = 1200
    height = 720

    # Frame rate
    fps = ""
    elapsedTime = 0

    # Video capturer
    cap = cv2.VideoCapture(vid_path)
    cv2.namedWindow('FPS', cv2.WINDOW_AUTOSIZE)

    # Image overlay
    overlay = np.zeros((input_shape[0], input_shape[1], 3), np.uint8)
    overlay[:] = (127, 0, 0)

    while True:

        # Read frames
        t1 = time.time()
        ret, frame = cap.read()

        # BGR->RGB, CV2->PIL
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        # Resize image
        image = image.resize(input_shape, Image.ANTIALIAS)

        # Normalization
        image = np.asarray(image)
        prepimg = image / 255.0
        prepimg = prepimg[np.newaxis, :, :, :]

        # Segmentation
        interpreter.set_tensor(
            input_details[0]['index'], np.array(prepimg, dtype=np.float32))
        interpreter.invoke()
        outputs = interpreter.get_tensor(output_details[0]['index'])

        # Process the output
        output = np.uint8(outputs[0] > 0.5)
        res = np.reshape(output, input_shape)
        mask = Image.fromarray(np.uint8(res), mode="P")
        mask = np.array(mask.convert("RGB")) * overlay
        mask = cv2.resize(np.asarray(mask), (width, height),
                          interpolation=cv2.INTER_CUBIC)
        frame = cv2.resize(frame, (width, height),
                           interpolation=cv2.INTER_CUBIC)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

        # Overlay the mask
        output = cv2.addWeighted(frame, 1, mask, 0.9, 0)

        # Display the output
        cv2.putText(output, fps, (width - 180, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('FPS', output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elapsedTime = time.time() - t1
        fps = "(Playback) {:.1f} FPS".format(1 / elapsedTime)
        print("fps = ", str(fps))

    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_parsed_cmd_args(
        default_model="models/transpose_seg/deconv_fin_munet.tflite")
    inference_model(args.source_vid_path,
                    args.model_path)


if __name__ == "__main__":
    main()
