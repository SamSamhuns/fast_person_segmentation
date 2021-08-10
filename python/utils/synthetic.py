import numpy as np
import argparse
import random
import cv2
import os


# Construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-iimg", "--input_image_path", required=True,
                help="path to source image directory")
ap.add_argument("-imsk", "--input_mask_path", required=True,
                help="path to  source mask directory")
ap.add_argument("-ibgd", "--input_background_path", required=True,
                help="path to background image directory")
ap.add_argument("-oimg", "--output_image_path", required=True,
                help="path to destination image directory")
ap.add_argument("-omsk", "--output_mask_path", required=True,
                help="path to destination mask image directory")
args = vars(ap.parse_args())

img_path = args["input_image_path"]
msk_path = args["input_mask_path"]

bgd_path = args["input_background_path"]

syn_img = args["output_image_path"]
syn_msk = args["output_mask_path"]

dirs_img = os.listdir(img_path)
dirs_img.sort()

dirs_msk = os.listdir(msk_path)
dirs_msk.sort()

dirs_bgd = os.listdir(bgd_path)
dirs_bgd.sort()

# Target size [modify if needed]
x, y = (256, 256)

# Ensure same name for corresponding mask and image


def resize():
    for item in dirs_img:
        if os.path.isfile(img_path + item):

            # Ensure masks are in png format
            png_msk = item.rsplit('.', 1)[0] + '.png'

            img = cv2.cvtColor(cv2.imread(img_path + item), cv2.COLOR_BGR2RGB)
            msk = cv2.cvtColor(cv2.imread(msk_path + png_msk), cv2.COLOR_BGR2RGB)
            bgd = cv2.cvtColor(cv2.imread(bgd_path + random.choice(dirs_bgd)), cv2.COLOR_BGR2RGB)

            # Resize foreground, mask and background
            img = cv2.resize(img, (x, y))
            msk = cv2.resize(msk, (x, y), cv2.INTER_NEAREST)
            bgd = cv2.resize(bgd, (x, y))

            # Perform alpha blending with new background
            img = np.array(img) / 255.0
            msk = np.array(msk)[:, :, 0].reshape(x, y, 1) / 255.0
            bgd = np.array(bgd) / 255.0

            synimg = bgd * (1.0 - msk) + img * msk

            # Save the new images and masks
            cv2.imwrite(syn_img + "syn_" + item, synimg)
            cv2.imwrite(syn_msk + "syn_" + png_msk, np.squeeze(msk))


if __name__ == "__main__":
    resize()

# Sample run : python synthetic_arg.py -iimg PNGImages_128/ -imsk PNGMasks_128/ -ibgd bgd_img/ -oimg syn_img/ -omsk syn_msk/
