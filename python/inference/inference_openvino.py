import cv2
import numpy as np
from time import time

# Import OpenVINO Inference Engine
from openvino.inference_engine import IECore
from utils.inference import load_bgd, get_cmd_argparser, get_config_dict, get_frame_after_postprocess
from utils.inference import PostProcessingType, ImageioVideoWriter, VideoStreamMultiThreadWidget


def get_openvino_core_net_exec(model_xml_path, model_bin_path, target_device="CPU"):
    # load IECore object
    OpenVinoIE = IECore()

    # load openVINO network
    OpenVinoNetwork = OpenVinoIE.read_network(
        model=model_xml_path, weights=model_bin_path)

    # create executable network
    OpenVinoExecutable = OpenVinoIE.load_network(
        network=OpenVinoNetwork, device_name=target_device)

    return OpenVinoIE, OpenVinoNetwork, OpenVinoExecutable


def inference_model(vid_path,
                    bg_img_path,
                    xml_model_path,
                    bin_model_path,
                    multi_thread=True,
                    json_config_path="models/model_info.json",
                    output_dir=None):
    # choose parameters
    post_processing = PostProcessingType.GAUSSIAN
    default_threshold = 0.63
    default_mopen_ksize = 7
    default_mopen_iter = 9
    default_gauss_ksize = 3
    bg_h, bg_w = 513, 513
    disp_h, disp_w = 720, 1200

    # Load Network and Executable
    OpenVinoIE, OpenVinoNetwork, OpenVinoExecutable = get_openvino_core_net_exec(
        xml_model_path, bin_model_path)

    # Get Input, Output Information
    InputLayer = next(iter(OpenVinoNetwork.input_info))
    OutputLayer = next(iter(OpenVinoNetwork.outputs))

    # load model config from json file
    config_dict = get_config_dict(xml_model_path, json_config_path)
    in_h, in_w = config_dict["in_height"], config_dict["in_width"]

    # Load background image, if path is None, use dark background
    bgd = load_bgd(bg_img_path, bg_w, bg_h)

    # load video
    vid_path = 0 if vid_path is None else vid_path

    cv2_disp_name = post_processing.name
    # check if multi-threading is to be used
    if multi_thread:
        cap = VideoStreamMultiThreadWidget(vid_path)
    else:
        cap = cv2.VideoCapture(vid_path)
    if output_dir is not None:
        vwriter = ImageioVideoWriter(output_dir, str(vid_path), __file__)

    ret, frame = cap.read()
    fps = ""
    while ret:
        t1 = time()
        # for handling multi_threading load
        ret, frame = cap.read()
        if frame is None:
            continue

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        simg = cv2.resize(img, (in_w, in_h),
                          interpolation=cv2.INTER_AREA) / 255.0
        simg = simg.transpose((2, 0, 1))
        simg = np.expand_dims(simg, axis=0)
        # Predict
        results = OpenVinoExecutable.infer(inputs={InputLayer: simg})
        out = results[OutputLayer][0]

        msk = np.float32(out).reshape((in_h, in_w, 1))
        """ MORPH_OPEN SMOOTHING """
        if post_processing == PostProcessingType.MORPH_OPEN:
            kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(default_mopen_ksize, default_mopen_ksize))
            msk = cv2.morphologyEx(msk,
                                   cv2.MORPH_OPEN,
                                   kernel=kernel,
                                   iterations=default_mopen_iter)

        """ GAUSSIAN SMOOTHING """
        if post_processing == PostProcessingType.GAUSSIAN:
            msk = cv2.GaussianBlur(msk,
                                   ksize=(default_gauss_ksize,
                                          default_gauss_ksize),
                                   sigmaX=4,
                                   sigmaY=0)
        # postprocess
        frame = get_frame_after_postprocess(
            msk, img, bgd, (bg_w, bg_h), (disp_w, disp_h), default_threshold)

        # Display the resulting frame & FPS
        cv2.putText(frame, fps, (disp_h - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(cv2_disp_name, frame[..., ::-1])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        vwriter.write_frame(frame) if output_dir else None
        fps = f"FPS: {1/(time() - t1):.1f}"
    vwriter.close() if output_dir else None
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = get_cmd_argparser(default_model=None)
    args = parser.parse_args()
    model_xml = "models/transpose_seg_openvino/deconv_bnoptimized_munet_e260_openvino/deconv_bnoptimized_munet_e260.xml"
    model_bin = "models/transpose_seg_openvino/deconv_bnoptimized_munet_e260_openvino/deconv_bnoptimized_munet_e260.bin"
    inference_model(vid_path=args.source_vid_path,
                    bg_img_path=args.bg_img_path,
                    xml_model_path=model_xml,
                    bin_model_path=model_bin,
                    multi_thread=args.use_multi_thread,
                    output_dir=args.output_dir)


if __name__ == "__main__":
    main()
