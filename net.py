import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
#         http://masc.cs.gmu.edu/wiki/partialconv
#         C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)
        
        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask


class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        elif sample=='pool':
            self.conv = nn.Sequential(PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias),nn.MaxPool2d(kernel_size=2,stride=2))
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

#         if bn:
#             self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
#         if hasattr(self, 'bn'):
#             h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask

class PConvUNet1(nn.Module):
    def __init__(self):
        super(PConvUNet1,self).__init__()
        self.conv1 = PartialConv(1,64,kernel_size=3,stride=1,padding=1, bias=False)
        self.conv2 = PartialConv(64,64,kernel_size=3,stride=1,padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size = 2,stride=2)
        self.conv3 = PartialConv(64,128,kernel_size=3,stride=1,padding=1, bias=False)
        self.conv4 = PartialConv(128,128,kernel_size=3,stride=1,padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size = 2,stride=2)
        self.conv5 = PartialConv(128,256,kernel_size=3,stride=1,padding=1, bias=False)
        self.conv6 = PartialConv(256,128,kernel_size=3,stride=1,padding=1, bias=False)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7 = PartialConv(256,128,kernel_size=3,stride=1,padding=1, bias=False)
        self.conv8 = PartialConv(128,64,kernel_size=3,stride=1,padding=1, bias=False)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv9 = PartialConv(128,64,kernel_size=3,stride=1,padding=1, bias=False)
        self.conv10 = PartialConv(64,64,kernel_size=3,stride=1,padding=1, bias=False)
        self.conv11 = nn.Conv2d(64,1,kernel_size=1,stride=1,padding=0)
        self.relu = nn.ReLU(True)
    def forward(self,inputs,masks):
        out1,mask1=self.conv1(inputs,masks)
        out1=self.relu(out1)
        out2,mask2=self.conv2(out1,mask1)
        out2=self.relu(out2)
        out3=self.pool1(out2)
        mask3=self.pool1(mask2)
        out4,mask4=self.conv3(out3,mask3)
        out4=self.relu(out4)
        out5,mask5=self.conv4(out4,mask4)
        out5=self.relu(out5)
        out6=self.pool2(out5)
        mask6=self.pool1(mask5)
        out7,mask7=self.conv5(out6,mask6)
        out7=self.relu(out7)
        out8,mask8=self.conv6(out7,mask7)
        out8=self.relu(out8)
        out9=self.up1(out8)
        mask9=self.up1(mask8)
        out9=torch.cat([out9,out5],dim=1)
        mask9=torch.cat([mask9,mask5],dim=1)
        out10,mask10=self.conv7(out9,mask9)
        out10=self.relu(out10)
        out11,mask11=self.conv8(out10,mask10)
        out11=self.relu(out11)
        out12=self.up2(out11)
        mask12=self.up2(mask11)
        out12=torch.cat([out12,out2],dim=1)
        mask12=torch.cat([mask12,mask2],dim=1)
        out13,mask13=self.conv9(out12,mask12)
        out13=self.relu(out13)
        out14,mask14=self.conv10(out13,mask13)
        out14=self.relu(out14)
        out15=self.conv11(out14)
        return out15,mask14
class PConvUNet(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512, 512, sample='down-3'))

        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512 + 512, 512, activ='leaky'))
        self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1 = PCBActiv(64 + input_channels, input_channels,
                              bn=False, activ=None, conv_bias=True)

    def forward(self, input, input_mask):
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        # concat upsampled output of h_enc_N-1 and dec_N+1, then do dec_N
        # (exception)
        #                            input         dec_2            dec_1
        #                            h_enc_7       h_enc_8          dec_8

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)

            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h_mask = F.interpolate(
                h_mask, scale_factor=2, mode='nearest')

            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)

        return h, h_mask
class Unet(nn.Module):
    def __init__(self, stride=1):
        super(Unet,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),nn.ReLU(True))
        self.pool1 = nn.MaxPool2d(kernel_size = 2,stride=2)
        self.conv3 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),nn.ReLU(True))
        self.pool2 = nn.MaxPool2d(kernel_size = 2,stride=2)
        self.conv5 = nn.Sequential(nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),nn.ReLU(True))
        self.conv6 = nn.Sequential(nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1),nn.ReLU(True))
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7 = nn.Sequential(nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1),nn.ReLU(True))
        self.conv8 = nn.Sequential(nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),nn.ReLU(True))
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv9 = nn.Sequential(nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),nn.ReLU(True))
        self.conv10 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),nn.ReLU(True))
        self.conv11 = nn.Sequential(nn.Conv2d(64,1,kernel_size=1,stride=1,padding=0))
    def forward(self,input):
        out1=self.conv1(input)
        out2=self.conv2(out1)
        out3=self.pool1(out2)
        out4=self.conv3(out3)
        out5=self.conv4(out4)
        out6=self.pool2(out5)
        out7=self.conv5(out6)
        out8=self.conv6(out7)
        out9=self.up1(out8)
        out9=torch.cat([out9,out5],dim=1)
        out10=self.conv7(out9)
        out11=self.conv8(out10)
        out12=self.up2(out11)
        out12=torch.cat([out12,out2],dim=1)
        out13=self.conv9(out12)
        out14=self.conv10(out13)
        out15=self.conv11(out14)
        return out15