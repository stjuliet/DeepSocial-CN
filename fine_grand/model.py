from __future__ import absolute_import, division, print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mymodels.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class ResNet_Decoder(nn.Module):
    def __init__(self, inplanes, bn_momentum=0.1):
        super(ResNet_Decoder, self).__init__()
        self.bn_momentum = bn_momentum
        self.inplanes = inplanes
        
        #----------------------------------------------------------#
        #   16,16,2048 -> 32,32,256 -> 64,64,128 -> 128,128,64
        #   利用ConvTranspose2d进行上采样。
        #   每次特征层的宽高变为原来的两倍。
        #----------------------------------------------------------#
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            num_filters=[256, 128, 64],
            num_kernels=[4, 4, 4],
        )

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    output_padding=0))
            layers.append(nn.BatchNorm2d(planes, momentum=self.bn_momentum))
            layers.append(nn.ReLU())
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)


class ResNet_Head(nn.Module):
    def __init__(self, num_classes=80, channel=64, bn_momentum=0.1):
        super(ResNet_Head, self).__init__()
        #-----------------------------------------------------------------#
        #   对获取到的特征进行上采样，进行分类预测和回归预测
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 2
        #-----------------------------------------------------------------#
        # 热力图预测部分
        self.act = nn.Sigmoid()
        # print('num_classes', num_classes)
        self.cls_head = nn.Sequential(
            nn.Conv2d(64, channel,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(channel, num_classes,
                      kernel_size=1, stride=1, padding=0))
        # 宽高预测的部分
        self.wh_head = nn.Sequential(
            nn.Conv2d(64, channel,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(channel, 2,
                      kernel_size=1, stride=1, padding=0))

        # 中心点预测的部分
        self.reg_head = nn.Sequential(
            nn.Conv2d(64, channel,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(channel, 2,
                      kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        hm = self.cls_head(x)
        # print(hm)
        wh = self.wh_head(x)
        offset = self.reg_head(x)
        return self.act(hm), wh, offset


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num=1, activ='sigmoid', num_bottleneck=512):
        super(ClassBlock, self).__init__()

        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]

        add_block = nn.Sequential(*add_block)
        #add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        if activ == 'sigmoid':
            classifier += [nn.Sigmoid()]
        elif activ == 'softmax':
            classifier += [nn.Softmax()]
        elif activ == 'none':
            classifier += []
        else:
            raise AssertionError("Unsupported activation: {}".format(activ))
        classifier = nn.Sequential(*classifier)
        #classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


class CenterNet_ResNet(nn.Module):
    def __init__(self, num_classes=20, config=None):
        super(CenterNet_ResNet, self).__init__()
        # self.n_class = config['n_class']
        # print(config['backbone'])
        # assert config['backbone'] in ['resnet18', 'resnet50_vd', 'resnet101_vd', 'resnet152_vd'], \
        #         'only support backbone of resnet_18, resnet50_vd, resnet101_vd, resnet152_vd'
        if config['backbone'] == 'resnet_18':
            self.backbone = resnet18(pretrained=config['pretrained'])
            last_dim = 512
        elif config['backbone'] == 'resnet34':
            self.backbone = resnet34(pretrained=config['pretrained'])
            last_dim = 512
        elif config['backbone'] == 'resnet50':
            self.backbone = resnet50(pretrained=config['pretrained'])
            last_dim = 2048
        elif config['backbone'] == 'resnet101':
            self.backbone = resnet101(pretrained=config['pretrained'])
            last_dim = 2048
        elif config['backbone'] == 'resnet152':
            self.backbone = resnet152(pretrained=config['pretrained'])
            last_dim = 2048
        else:
            print('only support backbone of resnet_18, resnet34, resnet50, resnet101, resnet152')

        self.decoder = ResNet_Decoder(last_dim)
        #-----------------------------------------------------------------#
        #   对获取到的特征进行上采样，进行分类预测和回归预测
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 2
        #-----------------------------------------------------------------#
        self.head = ResNet_Head(channel=64, num_classes=num_classes)

        # data_format="NCHW"
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.avg_pool_channels = last_dim
        stdv = 1.0 / math.sqrt(self.avg_pool_channels * 1.0)
        mapp = [1,3,3,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1]
        # mapp = [4,3,4,12,12]
        for c in range(21):
            self.__setattr__('class_%d' % c, ClassBlock(input_dim=self.avg_pool_channels, class_num=mapp[c], activ='sigmoid') )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for c in range(21):
            for param in self.__getattr__('class_%d' % c).parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        for c in range(21):
            for param in self.__getattr__('class_%d' % c).parameters():
                param.requires_grad = True

    def forward(self, x):
        feat = self.backbone(x) # n, c, h, w
        #print(feat.shape)
        x = self.avg_pool(feat)
        x = self.flatten(x)
        pred_label = [self.__getattr__('class_%d' % c)(x) for c in range(21)]
        pred_label = torch.cat(pred_label, 1)
        return self.head(self.decoder(feat)), pred_label
