import torch
import torch.nn as nn
from torchvision import models

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )


class ResNetUnetBlock(nn.Module):
    def __init__(self, n_channel_in, n_class_out):
        super(ResNetUnetBlock, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.conv1 = nn.Conv2d(n_channel_in, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up2 = convrelu(128 + 256, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(n_channel_in, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class_out, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        
        layer3 = self.layer3_1x1(layer3)
        x = self.upsample(layer3)

        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


class ResNetUNetHierarchical(nn.Module):
    def __init__(self, out1_n_class, out2_n_class, with_img_segm):
        super().__init__()

        self.with_img_segm = with_img_segm

        # takes 3 spatial classes, outputs 3 spatial classes
        self.unet1 = ResNetUnetBlock(n_channel_in=out1_n_class, n_class_out=out1_n_class)

        if with_img_segm:
            # 3 spatial classes + 1 coming from reduction of channels from object classes from img segmentation
            input_n_dim = out1_n_class + 1
            self.layer_imgSegm_in = convrelu(out2_n_class, 1, 1, 0)
        else:
            input_n_dim = out1_n_class

        self.unet2 = ResNetUnetBlock(n_channel_in=input_n_dim, n_class_out=out2_n_class)


    def forward(self, input, img_segm=None):
        B, T, C, cH, cW = input.shape
        input = input.view(B*T,C,cH,cW)

        out1 = self.unet1(input=input)

        if self.with_img_segm:
            B, T, C, cH, cW = img_segm.shape
            img_segm = img_segm.view(B*T,C,cH,cW)

            # reducing img segm channels from 27 to 1
            img_segm_in = self.layer_imgSegm_in(img_segm)
            
            input2 = torch.cat((out1, img_segm_in), dim=1)
            out2 = self.unet2(input=input2)

        else:
            out2 = self.unet2(input=out1)

        return out1, out2