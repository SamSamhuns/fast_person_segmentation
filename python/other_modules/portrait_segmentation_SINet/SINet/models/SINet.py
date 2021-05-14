
'''
ExtPortraitSeg
Copyright (c) 2019-present NAVER Corp.
MIT license
'''

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

BN_moment = 0.1


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)

        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(
            padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03, momentum=BN_moment)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class separableCBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)

        self.conv = nn.Sequential(
            nn.Conv2d(nIn, nIn, (kSize, kSize), stride=stride,
                      padding=(padding, padding), groups=nIn, bias=False),
            nn.Conv2d(nIn, nOut,  kernel_size=1, stride=1, bias=False),
        )
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03, momentum=BN_moment)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class SqueezeBlock(nn.Module):
    def __init__(self, exp_size, divide=4.0):
        super(SqueezeBlock, self).__init__()

        if divide > 1:
            self.dense = nn.Sequential(
                nn.Linear(exp_size, int(exp_size / divide)),
                nn.PReLU(int(exp_size / divide)),
                nn.Linear(int(exp_size / divide), exp_size),
                nn.PReLU(exp_size),
            )
        else:
            self.dense = nn.Sequential(
                nn.Linear(exp_size, exp_size),
                nn.PReLU(exp_size)
            )

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = torch.nn.functional.avg_pool2d(
            x, kernel_size=[height, width]).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)
        # out = hard_sigmoid(out)

        return out * x


class SEseparableCBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, divide=2.0):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)

        self.conv = nn.Sequential(
            nn.Conv2d(nIn, nIn, (kSize, kSize), stride=stride,
                      padding=(padding, padding), groups=nIn, bias=False),
            SqueezeBlock(nIn, divide=divide),
            nn.Conv2d(nIn, nOut,  kernel_size=1, stride=1, bias=False),
        )

        self.bn = nn.BatchNorm2d(nOut, eps=1e-03, momentum=BN_moment)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)

        output = self.bn(output)
        output = self.act(output)
        return output


class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''

    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03, momentum=BN_moment)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output


class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(
            padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03, momentum=BN_moment)

    def forward(self, input):
        '''

        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, group=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
                              padding=(padding, padding), bias=False, groups=group)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class S2block(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, config):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        kSize = config[0]
        avgsize = config[1]

        self.resolution_down = False
        if avgsize > 1:
            self.resolution_down = True
            self.down_res = nn.AvgPool2d(avgsize, avgsize)
            self.up_res = nn.UpsamplingBilinear2d(scale_factor=avgsize)
            self.avgsize = avgsize

        padding = int((kSize - 1) / 2)
        self.conv = nn.Sequential(
            nn.Conv2d(nIn, nIn, kernel_size=(kSize, kSize), stride=1,
                      padding=(padding, padding), groups=nIn, bias=False),
            nn.BatchNorm2d(nIn, eps=1e-03, momentum=BN_moment))

        self.act_conv1x1 = nn.Sequential(
            nn.PReLU(nIn),
            nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, bias=False),
        )

        self.bn = nn.BatchNorm2d(nOut, eps=1e-03, momentum=BN_moment)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        if self.resolution_down:
            input = self.down_res(input)
        output = self.conv(input)

        output = self.act_conv1x1(output)
        if self.resolution_down:
            output = self.up_res(output)
        return self.bn(output)


class S2module(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''

    def __init__(self, nIn, nOut, add=True, config=[[3, 1], [5, 1]]):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        print("This module has " + str(config))

        group_n = len(config)
        n = int(nOut / group_n)
        n1 = nOut - group_n * n

        self.c1 = C(nIn, n, 1, 1, group=group_n)
        # self.c1 = C(nIn, n, 1, 1)

        for i in range(group_n):
            var_name = 'd{}'.format(i + 1)
            if i == 0:
                self.__dict__["_modules"][var_name] = S2block(
                    n, n + n1, config[i])
            else:
                self.__dict__["_modules"][var_name] = S2block(n, n,  config[i])

        self.BR = BR(nOut)
        self.add = add
        self.group_n = group_n

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        output1 = channel_shuffle(output1, self.group_n)

        for i in range(self.group_n):
            var_name = 'd{}'.format(i + 1)
            result_d = self.__dict__["_modules"][var_name](output1)
            if i == 0:
                combine = result_d
            else:
                combine = torch.cat([combine, result_d], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.BR(combine)
        return output


class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''

    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            # pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(2, stride=2))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input


class SINet_Encoder(nn.Module):

    def __init__(self, config, classes=20, p=5, q=3,  chnn=1.0):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super().__init__()
        print("SINet Enc bracnch num :  " + str(len(config[0])))
        print("SINet Enc chnn num:  " + str(chnn))
        dim1 = 16
        dim2 = 48 + 4 * (chnn - 1)
        dim3 = 96 + 4 * (chnn - 1)

        self.level1 = CBR(3, 12, 3, 2)

        self.level2_0 = SEseparableCBR(12, dim1, 3, 2, divide=1)

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            if i == 0:
                self.level2.append(
                    S2module(dim1, dim2, config=config[i], add=False))
            else:
                self.level2.append(S2module(dim2, dim2, config=config[i]))
        self.BR2 = BR(dim2 + dim1)

        self.level3_0 = SEseparableCBR(dim2 + dim1, dim2, 3, 2, divide=2)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            if i == 0:
                self.level3.append(
                    S2module(dim2, dim3, config=config[2 + i], add=False))
            else:
                self.level3.append(S2module(dim3, dim3, config=config[2 + i]))
        self.BR3 = BR(dim3 + dim2)

        self.classifier = C(dim3 + dim2, classes, 1, 1)

    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        output1 = self.level1(input)  # 8h 8w

        output2_0 = self.level2_0(output1)  # 4h 4w

        # print(str(output1_0.size()))
        for i, layer in enumerate(self.level2):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)  # 2h 2w

        output3_0 = self.level3_0(
            self.BR2(torch.cat([output2_0, output2], 1)))  # h w
        # print(str(output2_0.size()))

        for i, layer in enumerate(self.level3):
            if i == 0:
                output3 = layer(output3_0)
            else:
                output3 = layer(output3)

        output3_cat = self.BR3(torch.cat([output3_0, output3], 1))
        classifier = self.classifier(output3_cat)
        return classifier


class SINet(nn.Module):

    def __init__(self, config, classes=20, p=2, q=3, chnn=1.0, encoderFile=None, pre_config=True):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        :param encoderFile: pretrained encoder weights. Recall that we first trained the ESPNet-C and then attached the
                            RUM-based light weight decoder. See paper for more details.
        '''
        super().__init__()
        print("SB Net Enc branch num :  " + str(len(config[0])))
        print("SB Net Enc chnn num:  " + str(chnn))
        dim1 = 16
        dim2 = 48 + 4 * (chnn - 1)
        dim3 = 96 + 4 * (chnn - 1)

        self.encoder = SINet_Encoder(config, classes, p, q, chnn)
        # # load the encoder modules
        if encoderFile != None:
            if torch.cuda.device_count() == 0:
                self.encoder.load_state_dict(
                    torch.load(encoderFile, map_location="cpu"))
            else:
                self.encoder.load_state_dict(torch.load(encoderFile))
            print('Encoder loaded!')

        # (scale_factor=2, mode='bilinear')
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.bn_3 = nn.BatchNorm2d(classes, eps=1e-03)
        self.level2_C = CBR(dim2, classes, 1, 1)
        self.bn_2 = nn.BatchNorm2d(classes, eps=1e-03)

        # self.classifier = nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False)
        self.classifier = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(classes, classes, 3, 1, 1, bias=False))

        # pre configuration settings for inference
        self.pre_config = pre_config
        if self.pre_config:
            # network image input conf
            self.imgW, self.imgH = 224, 224
            self.mean = [107.05183, 115.51994, 132.23213]
            self.std = [64.022835, 65.14661,  68.31369]

    def forward(self, input):
        '''
        :param input: RGB image
        :return: transformed feature map
        '''
        # if pre-confg is set then preprocess
        if self.pre_config:
            orig_h, orig_w = input.shape[:2]
            img = cv2.resize(input, (self.imgW, self.imgH))
            img = img.astype(np.float32)
            for j in range(3):
                img[:, :, j] -= self.mean[j]
            for j in range(3):
                img[:, :, j] /= self.std[j]

            img /= 255
            img = img.transpose((2, 0, 1))
            img_tensor = torch.from_numpy(img)
            input = torch.unsqueeze(img_tensor, 0)  # add a batch dim

        output1 = self.encoder.level1(input)  # 8h 8w
        output2_0 = self.encoder.level2_0(output1)  # 4h 4w

        for i, layer in enumerate(self.encoder.level2):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)  # 2h 2w

        output3_0 = self.encoder.level3_0(self.encoder.BR2(
            torch.cat([output2_0, output2], 1)))  # h w

        for i, layer in enumerate(self.encoder.level3):
            if i == 0:
                output3 = layer(output3_0)
            else:
                output3 = layer(output3)

        output3_cat = self.encoder.BR3(torch.cat([output3_0, output3], 1))
        Enc_final = self.encoder.classifier(output3_cat)  # 1/8

        Dnc_stage1 = self.bn_3(self.up(Enc_final))  # 1/4
        stage1_confidence = torch.max(nn.Softmax2d()(Dnc_stage1), dim=1)[0]
        b, c, h, w = Dnc_stage1.size()
        # TH = torch.mean(torch.median(stage1_confidence.view(b,-1),dim=1)[0])

        stage1_gate = (1 - stage1_confidence).unsqueeze(1).expand(b, c, h, w)

        Dnc_stage2_0 = self.level2_C(output2)  # 2h 2w
        Dnc_stage2 = self.bn_2(
            self.up(Dnc_stage2_0 * stage1_gate + (Dnc_stage1)))  # 4h 4w

        classifier = self.classifier(Dnc_stage2)

        # if pre-confg is set then postprocess
        if self.pre_config:
            img_out = torch.nn.functional.interpolate(
                classifier, (orig_h, orig_w), mode='bilinear')
            fg_mask = img_out[0].max(0)[1].byte().data.cpu().numpy()

            # only keep the largest connected component
            mask = np.uint8(fg_mask == 1)  # fg_mask only has 0 or 1 values
            labels, stats = cv2.connectedComponentsWithStats(mask, 4)[
                1:3]
            if len(stats[1:]) == 0: return fg_mask
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            # set all non-max connected components to 0
            fg_mask[labels != largest_label] = 0
            return fg_mask

        return classifier


def Enc_SINet(classes, p, q, chnn):
    # k, avg
    config = [[[3, 1], [5, 1]], [[3, 1], [3, 1]],
              [[3, 1], [5, 1]], [[3, 1], [3, 1]], [
                  [5, 1], [3, 2]], [[5, 2], [3, 4]],
              [[3, 1], [3, 1]], [[5, 1], [5, 1]], [[3, 2], [3, 4]], [[3, 1], [5, 2]]]

    model = SINet_Encoder(config, classes=classes, p=p, q=q, chnn=chnn)
    return model


def Dnc_SINet(classes=2, p=2, q=8, chnn=1, encoderFile=None, pre_config=True):
    config = [[[3, 1], [5, 1]], [[3, 1], [3, 1]],
              [[3, 1], [5, 1]], [[3, 1], [3, 1]], [
                  [5, 1], [3, 2]], [[5, 2], [3, 4]],
              [[3, 1], [3, 1]], [[5, 1], [5, 1]], [[3, 2], [3, 4]], [[3, 1], [5, 2]]]
    print("Dnc_SINet")

    model = SINet(config,
                  classes=classes, p=p, q=q,
                  chnn=chnn, encoderFile=encoderFile, pre_config=pre_config)

    ################# Preconfig Segments ######################
    if pre_config:
        config_weight_path = "./SINet/weight/SINet.pth"
        model.load_state_dict(torch.load(config_weight_path, "cpu"))

        # prune model
        # prune_conv2d = False
        # prune_linear = False
        # prune_batch_norm = False
        #
        # prune_method = prune.l1_unstructured
        #
        # for name, module in model.named_modules():
        #     # prune from connections in all 2D-conv layers
        #     if prune_conv2d and isinstance(module, torch.nn.Conv2d):
        #         prune_method(module, name='weight', amount=0.1)
        #     # prune from connections in all linear layers
        #     elif prune_linear and isinstance(module, torch.nn.Linear):
        #         prune_method(module, name='weight', amount=0.1)
        #         prune_method(module, name='bias', amount=0.01)
        #     # prune from connections in all BatchNorm layers
        #     elif prune_batch_norm and isinstance(module, torch.nn.BatchNorm2d):
        #         prune_method(module, name='weight', amount=0.1)

        # pytorch cpu quantization 'fbgemm' for server, 'qnnpack' for mobile
        # backend = 'fbgemm'
        # model.qconfig = torch.quantization.get_default_qat_qconfig(backend)

        model.eval()
    return model


if __name__ == "__main__":

    config = [[[3, 1], [5, 1]], [[3, 1], [3, 1]],
              [[3, 1], [5, 1]], [[3, 1], [3, 1]], [
                  [5, 1], [3, 2]], [[5, 2], [3, 4]],
              [[3, 1], [3, 1]], [[5, 1], [5, 1]], [[3, 2], [3, 4]], [[3, 1], [5, 2]]]

    model = SINet(classes=2, p=2, q=8, config=config,
                  chnn=1)

    # batch = torch.FloatTensor(1, 3, 480, 320)
    batch = torch.FloatTensor(1, 3, 224, 224)
    print(model(batch))
