import sys
import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

# Load the model
model = load_model('models/transpose_seg/deconv_bnoptimized_munet.h5')

# Load a test image
im = cv2.imread(sys.argv[1])

# Inference
im = cv2.resize(im, (128, 128))
img = np.float32(np.array(im) / 255.0)
img = img[:, :, 0:3]

# Reshape input and threshold output
out = model.predict(img.reshape(1, 128, 128, 3))
out = np.float32((out > 0.5)).reshape(128, 128, 1)

# Input image
plt.figure("Input")
plt.imshow(img)

# Output image
plt.figure("Output")
plt.imshow(out * img)

plt.show()
# Sample run: python test.py test/four.jpeg
