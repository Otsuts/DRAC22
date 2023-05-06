import timm
import clip
import torch
import torch.nn as nn
import torch.nn
from torchvision import models
import torchvision.models.resnet as resnet
from torchvision.models.convnext import convnext_base as convnext
import torchvision.models.vision_transformer as vit
import warnings
 
warnings.filterwarnings('ignore')


class SBConvNext(nn.Module):
    '''
    Convnext
    '''

    def __init__(self):
        super(SBConvNext, self).__init__()
        self.cnn = convnext(weights='ConvNeXt_Base_Weights.DEFAULT')
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1000, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 3),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(self.cnn(x))


class NFNet_f6(nn.Module):
    def __init__(self, model_name='beit_large_patch16_512', out_dim=3, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=out_dim)

    def get_input_shape(self):
        return self.model.default_cfg['input_size']

    def get_model_name(self):
        return self.model_name

    def forward(self, x):
        output = self.model(x)

        return output


class SBResNet(torch.nn.Module):
    '''
    Resnet
    '''

    def __init__(self):
        super(SBResNet, self).__init__()

        self.cnn = resnet.resnet101(weights='ResNet101_Weights.DEFAULT')
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.cnn.fc.weight.shape[0], 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.mlp(self.cnn(x))


class SBViT(torch.nn.Module):
    '''
    big vit
    '''

    def __init__(self):
        super(SBViT, self).__init__()
        # self.cnn = resnet.resnet101(weights='ResNet101_Weights.DEFAULT')
        self.vit = vit.vit_l_16(
            weights='ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1')
        # self.vit.heads.head= torch.nn.Linear(in_features=1024, out_features=3, bias=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1000, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.mlp(self.vit(x))


class SBViTB(torch.nn.Module):
    '''
    small vit
    '''

    def __init__(self):
        super(SBViTB, self).__init__()
        # self.cnn = resnet.resnet101(weights='ResNet101_Weights.DEFAULT')
        self.vit = vit.vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1')
        # self.vit.heads.head= torch.nn.Linear(in_features=768, out_features=3, bias=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1000, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.mlp(self.vit(x))


class SBResNeXt(torch.nn.Module):
    def __init__(self):
        '''
        效果略低于resnet
        '''
        super(SBResNeXt, self).__init__()
        self.resnxt = models.resnext101_64x4d(
            weights='ResNeXt101_64X4D_Weights.DEFAULT')
        self.resnxt.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=2048, out_features=256, bias=True),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(in_features=256, out_features=3, bias=True)
        )

    def forward(self, x):
        return self.resnxt(x)


class SBClip(torch.nn.Module):
    '''
    多模态语言模型
    '''

    def __init__(self):
        super(SBClip, self).__init__()
        # self.cnn = resnet.resnet101(weights='ResNet101_Weights.DEFAULT')
        self.vit = clip.load('ViT-B/32')
        # clip_model, _ = clip.load('vit..', jit=False)
        # self.vit.heads.head= torch.nn.Linear(in_features=768, out_features=3, bias=True)
        self.vit = self.vit[0].visual
        self.vit = self.vit.float()
        # print(self.vit)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 3),
        )
        # self.vit.ln_post= torch.nn.Linear(out_features=3, bias=True)
        # print(self.vit)

    def forward(self, x):
        return self.mlp(self.vit(x))


if __name__ == '__main__':
    sbconvnext = SBConvNext()