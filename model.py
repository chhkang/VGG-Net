import torch
import torch.nn as nn
import math

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            # 3,64
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 64,128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 128,256
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 256,512
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 512,512
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        #self.avg_pool = nn.AvgPool2d(3)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
        )

        #initial weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def vgg(data='cifar10'):
    if data == 'cifar10':
        return VGG16(num_classes=10)
    elif data == 'cifar100':
        return VGG16(num_classes=100)
