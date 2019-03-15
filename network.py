import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        use_bias = True
        self.delta = 0.01
        #   Convolution Block #1
        self.conv1_1 = nn.Conv2d(4, 64, kernel_size=3, padding=1,stride=1, bias=use_bias)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv1_bn = nn.BatchNorm2d(64,affine=True)
        self.down1=nn.Conv2d(64,64,kernel_size=1,stride=2,bias=False,padding=0)
        #   Convolution Block #2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv2_bn = nn.BatchNorm2d(128,affine=True)
        self.down2 = nn.Conv2d(128, 128, kernel_size=1, stride=2, bias=False, padding=0)
        #   Convolution Block #3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv3_bn = nn.BatchNorm2d(256,affine=True)
        self.down3 = nn.Conv2d(256, 256, kernel_size=1, stride=2, bias=False, padding=0)
        #   Convolution Block #4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv4_bn = nn.BatchNorm2d(512,affine=True)

        #   Convolution Block #5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=2, stride=1, dilation=2, bias=use_bias)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=2, stride=1, dilation=2, bias=use_bias)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=2, stride=1, dilation=2, bias=use_bias)
        self.conv5_bn = nn.BatchNorm2d(512,affine=True)

        #   Convolution Block #6
        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=3, padding=2, stride=1, dilation=2, bias=use_bias)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, padding=2, stride=1, dilation=2, bias=use_bias)
        self.conv6_3 = nn.Conv2d(512, 512, kernel_size=3, padding=2, stride=1, dilation=2, bias=use_bias)
        self.conv6_bn = nn.BatchNorm2d(512,affine=True)

        #   Convolution Block #7
        self.conv7_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv7_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv7_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv7_bn = nn.BatchNorm2d(512, affine=True)

        #   Short Cut Between Block #3 and Block #8
        self.conv3short8 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=use_bias)

        #   Convolution Block #8
        self.up8_1 = nn.ConvTranspose2d(512, 256, kernel_size=4, padding=1, stride=2, bias=use_bias)
        self.conv8_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv8_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv8_bn = nn.BatchNorm2d(256, affine=True)

        #   Short Cut Between Block #2 and Block #9
        self.conv2short9 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=use_bias)

        #   Convolution Block #9
        self.up9_1 = nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2, bias=use_bias)
        self.conv9_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv9_bn = nn.BatchNorm2d(128, affine=True)

        #   Short Cut Between Block #1 and Block #10
        self.conv1short10 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=use_bias)

        #   Convolution Block #10
        self.up10_1 = nn.ConvTranspose2d(128, 128, kernel_size=4, padding=1, stride=2, bias=use_bias)
        self.conv10_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv10_bn = nn.BatchNorm2d(128, affine=True)

        #   Output
        self.conv_out = nn.Conv2d(128, 2, kernel_size=3, padding=1, stride=1, bias=use_bias)



    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x_1 = self.conv1_bn(F.relu(self.conv1_2(x)))

        # downsampling between block #1, #2
        down_x = self.down1(x_1)
        #print(down_x.shape)

        x = F.relu(self.conv2_1(down_x))
        x_2 = self.conv2_bn(F.relu(self.conv2_2(x)))

        # downsampling between block #2, #3

        down_x = self.down2(x_2)
        #print(down_x.shape)

        x = F.relu(self.conv3_1(down_x))
        x = F.relu(self.conv3_2(x))
        x_3 = self.conv3_bn(F.relu(self.conv3_3(x)))

        # downsampling between block #3, #4
        down_x = self.down3(x_3)
        #print(down_x.shape)

        x = F.relu(self.conv4_1(down_x))
        x = F.relu(self.conv4_2(x))
        x = self.conv4_bn(F.relu(self.conv4_3(x)))

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = self.conv5_bn(F.relu(self.conv5_3(x)))
        #print(x.shape)

        x = F.relu(self.conv6_1(x))
        x = F.relu(self.conv6_2(x))
        x = self.conv6_bn(F.relu(self.conv6_3(x)))

        x = F.relu(self.conv7_1(x))
        x = F.relu(self.conv7_2(x))
        x = self.conv7_bn(F.relu(self.conv7_3(x)))

        #print(x.shape, x_3.shape)
        # upsampling between block #7, #8
        x = F.relu(self.up8_1(x)) + F.relu(self.conv3short8(x_3))
        x = F.relu(self.conv8_2(x))
        x = self.conv8_bn(F.relu(self.conv8_3(x)))

        # upsampling between block #8, #9
        x = F.relu(self.up9_1(x) + self.conv2short9(x_2))
        x = self.conv9_bn(F.relu(self.conv9_2(x)))

        # upsampling between block #9, #10
        x = F.relu(self.up10_1(x) + self.conv1short10(x_1))
        x = self.conv10_bn(F.relu(self.conv10_2(x)))

        x = torch.tanh(self.conv_out(x))

        return x
