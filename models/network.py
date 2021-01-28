import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torch.optim import lr_scheduler
import torch.nn.init as init
import math


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# from dataset import norm,denorm
from torchvision import transforms
normalise=transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
trans = transforms.Compose([normalise])

def calc_mean_std(features):
    """
    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """

    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std

def adain(content_features, style_features):
    """
    Adaptive Instance Normalization

    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features


class RC(nn.Module):
    """A wrapper of ReflectionPad2d and Conv2d"""
    def __init__(self, in_channels, out_channels, kernel_size=3, pad_size=1, activated=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.activated = activated

    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)
        if self.activated:
            return F.relu(h)
        else:
            return h

class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.slice1 = vgg[:2]
        self.nopool1 = vgg[2:4]
        self.slice2 = vgg[4:7]
        self.nopool2=vgg[7:9]
        self.slice3 = vgg[9:12]
        self.nopool3=vgg[12:18]
        self.slice4 = vgg[18:21] #relu4_1
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images, output_last_feature=False):
        h1 = self.slice1(images)
        np1=self.nopool1(h1)
        h2 = self.slice2(np1)
        np2=self.nopool2(h2)
        h3 = self.slice3(np2)
        np3=self.nopool3(h3)
        h4 = self.slice4(np3)
        if output_last_feature:
            return h4,np3,np2,np1
        else:
            return h1, h2, h3, h4

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rc1 = RC(512, 256, 3, 1)
        self.rc2 = RC(256, 256, 3, 1)
        self.rc3 = RC(256, 256, 3, 1)
        self.rc4 = RC(256, 256, 3, 1)
        self.rc5 = RC(256, 128, 3, 1)
        self.rc6 = RC(128, 128, 3, 1)
        self.rc7 = RC(128, 64, 3, 1)
        self.rc8 = RC(64, 64, 3, 1)
        self.rc9 = RC(64, 3, 3, 1, False)
        self.unpooling1=nn.ConvTranspose2d(256,256,kernel_size=2, stride=2, padding=0, bias=False,groups=256)
        self.unpooling2=nn.ConvTranspose2d(128,128,kernel_size=2, stride=2, padding=0, bias=False,groups=128)
        self.unpooling3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0, bias=False, groups=64)
    def forward(self, features,np3,np2,np1):
        h = self.rc1(features)
        h=self.unpooling1(h)
        h=np3+h
        h = self.rc2(h)
        h = self.rc3(h)
        h = self.rc4(h)
        h = self.rc5(h)
        h = self.unpooling2(h)
        h=np2+h
        h = self.rc6(h)
        h = self.rc7(h)
        h = self.unpooling3(h)
        h=h+np1
        h = self.rc8(h)
        h = self.rc9(h)
        return h
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_encoder = VGGEncoder()
        self.decoder = Decoder()
    def generate(self, content_images, style_images):
        content_features,np3,np2,np1 = self.vgg_encoder(content_images, output_last_feature=True)
        style_features ,sp3,sp2,sp1= self.vgg_encoder(style_images, output_last_feature=True)
        t = adain(content_features, style_features)
        csp3=adain(np3,sp3)
        csp2 = adain(np2, sp2)
        csp1 = adain(np1, sp1)
        out = self.decoder(t,csp3,csp2,csp1)
        return out
