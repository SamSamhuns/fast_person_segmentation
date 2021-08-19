from cv_client.selfie_segmentation import SelfieSegmentation, VideoStreamMT
from cv_client.selfie_segmentation import install_pip_package
from flask import Flask, Response, render_template
from time import time, sleep
import threading
import argparse
import datetime
import cv2

# set to True to use external raspberry camera
use_picam = False

# init the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (when mult browsers/tabs are viewing the stream)
output_frame = None
lock = threading.Lock()
app = Flask(__name__)


# init the video stream and allow camera sensor warmup
if use_picam:
    install_pip_package("imutils")
    import imutils
    vstream = imutils.VideoStream(usePiCamera=1).start()
else:
    vstream = VideoStreamMT(src=0)
sleep(2.0)


def segment_video_stream(bg_img_path=None,
                         tflite_model_path="cv_client/selfie_segmentation/weights/model_float16_quant.tflite"):
    """
    loops over frames from video stream
    segments person silhouettes
    draws results on the output_frame
    """
    global vstream, output_frame, lock
    segmentor = SelfieSegmentation(bg_img_path, tflite_model_path)
    ret, frame = vstream.read()
    fh, fw = frame.shape[:2]
    disp_w = 1280
    disp_h = int((fh * disp_w) / fw)
    disp_wh = (disp_w, disp_h)
    fps = ""

    while True:
        t1 = time()
        ret, frame = vstream.read()
        # rsz with aspect
        frame = cv2.resize(frame, disp_wh)
        # detect person in frame
        frame = segmentor.segment_frame(frame, disp_wh)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # draw info on frame
        cv2.putText(frame, fps, (disp_h - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        # acquire the lock, set the output frame, and release the lock
        # ensures output_frame is not written on while its being read
        with lock:
            output_frame = frame.copy()
        fps = f"FPS: {1/(time() - t1):.1f}"


def generate():
    """
    encode output_frame as JPEG data
    """
    global output_frame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available,
            # otherwise skip loop iteration
            if output_frame is None:
                continue
            flag, encodedImage = cv2.imencode(".jpg", output_frame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


@app.route("/")
def index():
    return render_template("index_local_cv2.html")


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Flask webserver with socketIO")
    parser.add_argument("-i", "--ip", type=str, default="0.0.0.0",
                        help="ip address of the device. Default: 0.0.0.0")
    parser.add_argument("-o", "--port", type=int, default=8080,
                        help="ephemeral port number of the server (1024 to 65535). Default: 8000")
    parser.add_argument("-bg", "--bgd_img_path", type=str,
                        help="background image path. If not provided, use a dark background")
    args = parser.parse_args()
    # start a thread that will perform person segmentation
    t = threading.Thread(target=segment_video_stream, args=(args.bgd_img_path,))
    t.daemon = True
    t.start()
    # start the flask app
    app.run(host=args.ip, port=args.port, debug=True,
            threaded=True, use_reloader=False)

# release the video stream pointer
vstream.release()
