import sys

import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


# Load a trained model checkpoint
model = load_model(sys.argv[1])

# compile model
model.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-3), metrics=["accuracy"])

# Load a test dataset (UINT8)
new_xtest = np.load("data/img_uint8.npy")
new_ytest = np.load("data/msk_uint8.npy")

# Evaluate model
score = model.evaluate(
    np.float32(new_xtest / 255.0), np.float32(new_ytest / 255.0), verbose=0
)

# Print loss and accuracy
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Sample run: python eval.py checkpoints/model.hdf5
