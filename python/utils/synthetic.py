from imageio import imsave
from PIL import Image
import numpy as np
import argparse
import random
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

            img = Image.open(img_path + item).convert('RGB')
            msk = Image.open(msk_path + png_msk).convert('RGB')
            bgd = Image.open(bgd_path + random.choice(dirs_bgd)).convert('RGB')

            # Resize foreground, mask and background
            img = img.resize((x, y), Image.ANTIALIAS)
            msk = msk.resize((x, y), Image.NEAREST)
            bgd = bgd.resize((x, y), Image.ANTIALIAS)

            # Perform alpha blending with new background
            img = np.array(img) / 255.0
            msk = np.array(msk)[:, :, 0].reshape(x, y, 1) / 255.0
            bgd = np.array(bgd) / 255.0

            synimg = bgd * (1.0 - msk) + img * msk

            # Save the new images and masks
            imsave(syn_img + "syn_" + item, synimg)
            imsave(syn_msk + "syn_" + png_msk, np.squeeze(msk))


if __name__ == "__main__":
    resize()

# Sample run : python synthetic_arg.py -iimg PNGImages_128/ -imsk PNGMasks_128/ -ibgd bgd_img/ -oimg syn_img/ -omsk syn_msk/
