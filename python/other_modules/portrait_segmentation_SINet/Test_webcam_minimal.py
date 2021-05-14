'''
ExtPortraitSeg
Copyright (c) 2019-present NAVER Corp.
MIT license
'''

import cv2
import json
import torch
import numpy as np
from time import time

import SINet.models as models

if __name__ == '__main__':
    skip_frame = 2  # how many frames to skip before calculating mask
    model = models.Dnc_SINet()

    ############### cv2 video testing ##########################################
    cap = cv2.VideoCapture(0)
    # result = cv2.VideoWriter('output_recording.mp4',
    #                          cv2.VideoWriter_fourcc(*'X264'),
    #                          25, (int(cap.get(3)), int(cap.get(4))))

    FRAME_COUNT = 0  # should always start with 0
    while(cap.isOpened()):
        t1 = time()
        ret, frame = cap.read()

        if ret:
            img_orig = frame

            if FRAME_COUNT % skip_frame == 0:
                FRAME_COUNT = 0
                with torch.no_grad():
                    fg_mask = model(frame)

            FRAME_COUNT += 1
            img_orig[fg_mask == 0, :] = 0

            cv2.imshow('frame', img_orig)
            # result.write(img_orig)
            print("fps: ", 1 / (time() - t1))
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # result.release()
    cap.release()
    cv2.destroyAllWindows()
