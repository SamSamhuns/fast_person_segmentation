import numpy as np
import matplotlib.pyplot as plt


def plot_fps_arr(data_dict_list, title, max_frames=1000, savepath=None):
    """
    data_dict_list is a list of dicts
    The dict must contain one key "x" which represents the arr data as a list/numpy array,
    additonal params related to plt.show() can be sent as required
    """
    print(title)
    plt.figure(figsize=(18, 10))

    for kwargs in data_dict_list:
        x = kwargs["x"][:max_frames]
        del kwargs["x"]

        avg_fps = np.mean(x)
        if "label" in kwargs:
            print(f"{kwargs['label']} Average FPS: {avg_fps}")
        plt.axhline(y=avg_fps, color="black", linestyle="dashdot")
        plt.text(-112, avg_fps, f"AVG:{avg_fps:.1f}", fontsize=12, style="oblique")
        plt.plot(x, **kwargs)
    plt.title(title + " Segmentation FPS", fontsize=20)
    plt.ylabel("FPS", fontsize=18)
    plt.xlabel("Frames", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="lower right", prop={"size": 20})
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    plt.clf()


def main():
    m2_tf_pb = [
        {
            "x": np.load("python_npy_and_graph/m2_tf_pb.npy"),
            "color": "b",
            "linewidth": 1.5,
            "label": "TF pb",
            "linestyle": "dashed",
            "alpha": 0.7,
        },
        {
            "x": np.load("python_npy_and_graph/m2_tf_pb_avx2_fma.npy"),
            "color": "r",
            "linewidth": 1.5,
            "label": "TF pb + AVX2,FMA",
            "linestyle": None,
            "alpha": 0.7,
        },
        {
            "x": np.load("python_npy_and_graph/m2_tf_pb_avx2_fma.npy"),
            "color": "g",
            "linewidth": 1.5,
            "label": "TF pb + AVX2,FMA + threading",
            "linestyle": "dotted",
            "alpha": 0.7,
        },
    ]
    plot_fps_arr(
        m2_tf_pb,
        title="Person Segmentation FPS with TF ProtoBuf mobilenet-v2",
        savepath="python_npy_and_graph/m2_tf_pb.jpg",
    )

    m3_tf_pb = [
        {
            "x": np.load("python_npy_and_graph/m3_tf_pb.npy"),
            "color": "b",
            "linewidth": 1.5,
            "label": "TF pb",
            "linestyle": "dashed",
            "alpha": 0.7,
        },
        {
            "x": np.load("python_npy_and_graph/m3_tf_pb_avx2_fma.npy"),
            "color": "r",
            "linewidth": 1.5,
            "label": "TF pb + AVX2,FMA",
            "linestyle": None,
            "alpha": 0.7,
        },
        {
            "x": np.load("python_npy_and_graph/m3_tf_pb_avx2_fma.npy"),
            "color": "g",
            "linewidth": 1.5,
            "label": "TF pb + AVX2,FMA + threading",
            "linestyle": "dotted",
            "alpha": 0.7,
        },
    ]
    plot_fps_arr(
        m3_tf_pb,
        title="Person Segmentation FPS with TF ProtoBuf mobilenet-v3",
        savepath="python_npy_and_graph/m3_tf_pb.jpg",
    )

    m3_ov = [
        {
            "x": np.load("python_npy_and_graph/m3_ov.npy"),
            "color": "b",
            "linewidth": 1.5,
            "label": "OpenVINO",
            "linestyle": "dashed",
            "alpha": 0.7,
        },
        {
            "x": np.load("python_npy_and_graph/m3_ov_mt.npy"),
            "color": "r",
            "linewidth": 1.5,
            "label": "OpenVINO + threading",
            "linestyle": None,
            "alpha": 0.7,
        },
    ]
    plot_fps_arr(
        m3_ov,
        title="Person Segmentation FPS with OpenVINO mobilenet-v3",
        savepath="python_npy_and_graph/m3_ov.jpg",
    )

    m3_tflite = [
        {
            "x": np.load("python_npy_and_graph/m3_tflite.npy"),
            "color": "b",
            "linewidth": 1.5,
            "label": "TF-Lite",
            "linestyle": "dashed",
            "alpha": 0.7,
        },
        {
            "x": np.load("python_npy_and_graph/m3_tflite_xnnpack.npy"),
            "color": "r",
            "linewidth": 1.5,
            "label": "TF-Lite + XNNPack",
            "linestyle": None,
            "alpha": 0.7,
        },
        {
            "x": np.load("python_npy_and_graph/m3_tflite_xnnpack_mt.npy"),
            "color": "g",
            "linewidth": 1.5,
            "label": "TF-Lite + XNNPack + threading",
            "linestyle": "dotted",
            "alpha": 0.7,
        },
    ]
    plot_fps_arr(
        m3_tflite,
        title="Person Segmentation FPS with TF-Lite mobilenet-v3",
        savepath="python_npy_and_graph/m3_tflite.jpg",
    )


if __name__ == "__main__":
    main()
