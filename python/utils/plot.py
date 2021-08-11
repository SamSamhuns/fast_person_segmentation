import numpy as np
import matplotlib.pyplot as plt


def plot_fps_arr(data_dict_list, title,  savepath=None):
    """
    data_dict_list is a list of dicts
    The dict must contain one key "x" which represents the arr data,
    additonal params related to plt.show() can be sent as required
    """
    plt.figure(figsize=(12, 10))

    for kwargs in data_dict_list:
        x = kwargs['x']
        del kwargs['x']
        plt.plot(x, **kwargs)
    plt.title(title + " Segmentation FPS", fontsize=20)
    plt.ylabel("FPS", fontsize=18)
    plt.xlabel("Frames", fontsize=18)
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    plt.clf()


def main():
    tf_graphdef_data = [{'x': np.load("orig_pb.npy"), 'color': 'b', 'label': 'Original',
                         'linestyle': 'dashed', 'alpha': 0.7},
                        {'x': np.load("multit_pb.npy"), 'color': 'r', 'label': 'Improved'}]
    plot_fps_arr(tf_graphdef_data, title="TFGraphDef",  savepath="tf_graphdef_fps.png")

    tf_graphdef_data = [{'x': np.load("orig_mp.npy"), 'color': 'b', 'label': 'Original',
                         'linestyle': 'dashed', 'alpha': 0.7},
                        {'x': np.load("multit_mp.npy"), 'color': 'r', 'label': 'Improved'}]
    plot_fps_arr(tf_graphdef_data, title="MediaPipe",  savepath="mediapipe_fps.png")


if __name__ == "__main__":
    main()
