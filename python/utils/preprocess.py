import numpy as np


# Preprocessing functions (runtime)
def calc_mean_std(img_arr):
    """img_arr must have shape [b_size, width, height, channel]
    """
    nimages, mean, std = img_arr.shape[0], 0., 0.
    # Rearrange batch to be the shape of [B, C, W * H]
    img_arr = img_arr.reshape(nimages, img_arr.shape[-1], -1)
    # Compute mean and std here
    mean = img_arr.mean(2).sum(0) / nimages
    std = img_arr.std(2).sum(0) / nimages

    return mean / 255, std / 255


def normalize_batch(imgs,
                    mean=[0.50693673, 0.47721124, 0.44640532],
                    std=[0.28926975, 0.27801928, 0.28596011]):
    """
    mean & std values are for default dataset
    """
    if imgs.shape[-1] > 1:
        return (imgs - np.array(mean)) / np.array(std)
    else:
        return imgs.round()


def denormalize_batch(imgs, should_clip=True,
                      mean=[0.50693673, 0.47721124, 0.44640532],
                      std=[0.28926975, 0.27801928, 0.28596011]):
    """
    mean & std values are for default dataset
    """
    imgs = (imgs * np.array(mean)
            ) + np.array(std)

    if should_clip:
        imgs = np.clip(imgs, 0, 1)
    return imgs
