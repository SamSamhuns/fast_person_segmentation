import numpy as np
import cv2
import os

'''
Images: SxS
Values: 0-255
Format: (PNG or JPEG) RGB
'''

# Convert source images into '.npy' format
final_img_dimen = (128, 128)

# img_path = "/path/to/image/dir/" # sample path to training img dir
img_path = "data/VOCdevkit/person/JPEGImagesOUT/"  # Pascal VOC
# img_path = "data/ai_segment/clip_img_OUT/"  # AI SEGMENT
img_dirs = os.listdir(img_path)
img_dirs.sort()
x_train = []


def load_image():
    for item in img_dirs:
        if os.path.isfile(img_path + item):
            im = cv2.cvtColor(cv2.imread(img_path + item), cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, final_img_dimen)
            im = im.astype('uint8')
            x_train.append(im)


load_image()

imgset = np.array(x_train)
print("Shape of training original data", imgset.shape)
np.save("data/voc_img_uint8.npy", imgset)


'''
Masks: SxS
Values: 0 (background) and 255 (foreground)
Format: PNG (RGB or ALPHA)
'''

# Convert mask images into '.npy' format

# msk_path = "/path/to/mask/dir/" # sample path to training img dir
msk_path = "data/VOCdevkit/person/SegmentationClassOUT/"  # Pascal VOC
# msk_path = "data/ai_segment/matting_OUT/"  # AI SEGMENT
msk_dirs = os.listdir(msk_path)
msk_dirs.sort()
y_train = []


def load_mask():
    for item in msk_dirs:
        if os.path.isfile(msk_path + item):
            im = cv2.cvtColor(cv2.imread(msk_path + item), cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, final_img_dimen)
            im = im[..., 0].astype('uint8')
            # removing non-binary values introduced after resizing
            im = np.where(im < 250, 0, 255)
            im = im.astype('uint8')
            y_train.append(im)


load_mask()

mskset = np.array(y_train)
print("Shape of training segmented data", mskset.shape)
np.save("data/voc_msk_uint8.npy", mskset[..., np.newaxis])
