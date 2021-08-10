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
import torch.nn.utils.prune as prune

if __name__ == '__main__':
    # these configurations have been preset in the SINet module
    config_weight_path = "./SINet/weight/SINet.pth"
    config_model = "Dnc_SINet"
    config_num_classes = 2
    config_p = 2
    config_q = 8
    config_chnn = 1

    skip_frame = 1  # how many frames to skip before calculating mask
    skip_frame += 1

    # pre_config here is set to false as to set conf manually
    model = models.__dict__[config_model](
        classes=config_num_classes,
        p=config_p,
        q=config_q,
        chnn=config_chnn,
        pre_config=False)

    model.load_state_dict(torch.load(config_weight_path, "cpu"))

    # prune model
    prune_conv2d = True
    prune_linear = True
    prune_batch_norm = False

    prune_method = prune.l1_unstructured

    for name, module in model.named_modules():
        # prune from connections in all 2D-conv layers
        if prune_conv2d and isinstance(module, torch.nn.Conv2d):
            prune_method(module, name='weight', amount=0.1)
        # prune from connections in all linear layers
        elif prune_linear and isinstance(module, torch.nn.Linear):
            prune_method(module, name='weight', amount=0.1)
            prune_method(module, name='bias', amount=0.01)
        # prune from connections in all BatchNorm layers
        elif prune_batch_norm and isinstance(module, torch.nn.BatchNorm2d):
            prune_method(module, name='weight', amount=0.1)

    # pytorch cpu quantization 'fbgemm' for server, 'qnnpack' for mobile
    backend = 'fbgemm'
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)

    # network image input conf
    imgW, imgH = 224, 224
    mean = [107.05183, 115.51994, 132.23213]
    std = [64.022835, 65.14661,  68.31369]

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
            img = frame
            (h, w) = img.shape[:2]
            img_orig = img

            if FRAME_COUNT % skip_frame == 0:
                FRAME_COUNT = 0
                img = cv2.resize(img, (imgW, imgH))
                img = img.astype(np.float32)
                for j in range(3):
                    img[:, :, j] -= mean[j]
                for j in range(3):
                    img[:, :, j] /= std[j]

                img /= 255
                img = img.transpose((2, 0, 1))
                img_tensor = torch.from_numpy(img)
                img_tensor = torch.unsqueeze(
                    img_tensor, 0)  # add a batch dimension

                with torch.no_grad():
                    img_out = model(img_tensor)

                img_out = torch.nn.functional.interpolate(
                    img_out, (h, w), mode='bilinear')
                fg_mask = img_out[0].max(0)[1].byte().data.cpu().numpy()
                # Alternative way to resize with opencv
                # fg_mask = cv2.resize(fg_mask, (w, h),
                #                 interpolation=cv2.INTER_LINEAR)

                # only keep the largest connected component
                mask = np.uint8(fg_mask == 1)  # fg_mask only has 0 or 1 values
                labels, stats = cv2.connectedComponentsWithStats(mask, 4)[
                    1:3]
                if len(stats[1:]) != 0:
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    # set all non-max connected components to 0
                    fg_mask[labels != largest_label] = 0

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
