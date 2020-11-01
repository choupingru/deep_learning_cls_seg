import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features    
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # outputs = []
        # for index, layer in enumerate(self.features):
        #     x = layer(x)
        #     if index in [22, 29, 30]:
        #         outputs.append(x)

        # return outputs
        return self.features(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        vgg_state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)

        yolo_state_dict = model.state_dict()
        for k in vgg_state_dict.keys():
            if k in yolo_state_dict.keys() and k.startswith('features'):
                yolo_state_dict[k] = vgg_state_dict[k]
        print('Loading Weight Success')
        model.load_state_dict(yolo_state_dict)
    return model


def vgg16(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)



class model_vgg16(nn.Module):

    def __init__(self, num_classes=50):
        super().__init__()
        
        self.fc1 = nn.Sequential(
            nn.Conv2d(512, 512, 5, 1, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        self.fc3 = nn.Sequential(
            nn.Conv2d(512, 7, 1, 1),
        )
        self.pred1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Upsample(scale_factor=2),
            nn.Upsample(scale_factor=8)

        )
        self._init_weight()
        self.vgg = vgg16(pretrained=True)
        #self.vgg.eval()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        
    def forward(self, input):
        #with torch.no_grad():
        y512 = self.vgg(input)
        # a, b, c, d = input.size()
        y512 = self.fc1(y512)
        y512 = self.fc2(y512)
        y512 = self.fc3(y512)
        pred = self.pred1(y512)

        # predx512 = self.pred2(x512)
        # predx256 = self.pred3(x256)

        # pred = F.interpolate(predx256, size=(512, 512)) + F.interpolate(predx512, size=(512, 512)) + F.interpolate(predy, size=(512, 512))

        b, c, x, y = pred.size()
        # print(pred.size())
        pred = pred.view(b, c, -1)
        pred = pred.permute(0, 2, 1).contiguous()
        pred = pred.view(-1, 7)

        return pred


def get_model():
    return model_vgg16()
    

if __name__ == '__main__':

    net = model_vgg16()
    print(net)
    x = torch.zeros((1, 3, 32, 32))
    print(net(x))












