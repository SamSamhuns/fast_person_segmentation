import os
import cv2
import glob

# Configure input and output directory paths
source_mask_list = glob.glob("/path/to/source/Labels/*.*", recursive=True)
target_mask_path = "/path/to/target/Labels/"

# Set foreground pixel value and target image size
fgd_pixval = 255
tgt_size = (256, 256)

print(source_mask_list)


def remask():
    # Convert and save the binary masks
    image_count = 0
    for item in source_mask_list:
        if os.path.isfile(item):
            im = cv2.imread(item)
            # Convert the mask to binary alpha
            im = im.split()[-1]
            im = cv2.resize(im, (tgt_size), cv2.INTER_NEAREST)

            # Threshold the image using value 127
            im[im >= 127] = fgd_pixval
            im[im < 127] = 0

            # Save the alpha mask
            file_name = os.path.basename(item)
            cv2.imwrite(target_mask_path + file_name, im)
            image_count = image_count + 1
            print("Processing: " + str(image_count))


if __name__ == "__main__":
    remask()
