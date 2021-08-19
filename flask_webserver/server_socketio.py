import io
import cv2
import base64
import argparse
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from cv_client.selfie_segmentation import SelfieSegmentation


app = Flask(__name__)
socketio = SocketIO(app)
segmentor = SelfieSegmentation(tflite_model_path="cv_client/selfie_segmentation/weights/model_float16_quant.tflite")


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index_socketio.html')


@socketio.on('image')
def image(data_image):
    sbuf = io.StringIO()
    sbuf.write(data_image)

    # decode and convert into image in BGR space
    img_stream = io.BytesIO(base64.b64decode(data_image))
    frame = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)

    # Process the image frame
    disp_wh = (500, 375)
    frame = segmentor.segment_frame(frame, disp_wh)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    imgencode = cv2.imencode('.jpg', frame)[1]
    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Flask webserver with socketIO")
    parser.add_argument("-i", "--ip", type=str, default="0.0.0.0",
                        help="ip address of the device. Default: 0.0.0.0")
    parser.add_argument("-o", "--port", type=int, default=8080,
                        help="ephemeral port number of the server (1024 to 65535). Default: 8000")
    parser.add_argument("-bg", "--bgd_img_path", type=str,
                        help="background image path. If not provided, use a dark background")
    args = parser.parse_args()
    segmentor.load_new_bgd(args.bgd_img_path)
    socketio.run(app=app, host=args.ip, port=args.port)
