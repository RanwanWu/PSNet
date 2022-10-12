import torch
import torch.nn as nn
import torchvision.models as models
from ResNet import ResNet50
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self, channel):
        super(Attention, self).__init__()

        self.conv1 = nn.Conv2d(channel, channel // 2, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(channel, channel // 2, 1, 1, 0, bias=False)
        self.conv3 = nn.Conv2d(channel // 2, channel // 2, 3, 1, 1, bias=False)
        self.conv4 = nn.Conv2d(channel, channel, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(channel // 2, channel // 2 , 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(channel // 2, channel // 2 , 3, 1, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(channel // 2)
        self.bn2 = nn.BatchNorm2d(channel // 2)
        self.bn3 = nn.BatchNorm2d(channel // 2)
        self.bn4 = nn.BatchNorm2d(channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.relu = nn.ReLU()

    def forward(self, t, x):
        x = self.relu(self.bn1(self.conv1(x)))
        t = self.relu(self.bn2(self.conv2(t)))
        fc = x + t
        fc = self.relu(self.bn3(self.conv3(fc)))
        fc = self.relu(self.conv3(fc))
        t = self.maxpool(self.relu(self.conv5(t)))
        x = self.maxpool(self.relu(self.conv6(x)))
        t = torch.mul(fc, t)
        x = torch.mul(fc, x)
        out = torch.cat((x, t), 1)
        f = out
        out = self.avgpool(self.relu(self.bn4(self.conv4(out))))
        out = torch.mul(out, f)
        return out

#Finally Transposed Module
class FTM(nn.Module):
    def __init__(self, channel):
        super(FTM, self).__init__()
        self.conv1 = nn.Conv2d(4*channel, 4*channel, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(4*channel, 3*channel, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(3*channel, 3*channel, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(3*channel, 2*channel, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(2*channel, 2*channel, kernel_size=1, stride=1, padding=0)
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(4*channel, 3*channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3*channel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(3*channel, 3*channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3*channel),
            nn.ReLU(inplace=True)
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(3*channel, 2*channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2*channel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2*channel, 2*channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2*channel),
            nn.ReLU(inplace=True)
        )

        self.bn1 = nn.BatchNorm2d(4*channel)
        self.bn2 = nn.BatchNorm2d(3*channel)
        self.bn3 = nn.BatchNorm2d(3*channel)
        self.bn4 = nn.BatchNorm2d(2*channel)
        self.bn5 = nn.BatchNorm2d(2*channel)
        self.relu = nn.ReLU(inplace=True)

        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x + self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x))) + self.deconv_1(x)
        x = x + self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x))) + self.deconv_2(x)
        x = x.mul(self.sigmoid(self.relu(self.bn5(self.conv5(x)))))
        x = self.upsample2(x)
        return x

#BBSNet
class BBSNet(nn.Module):
    def __init__(self):
        super(BBSNet, self).__init__()
        
        #Backbone model
        self.resnet = ResNet50('rgb')
        self.resnet_depth=ResNet50('rgbd')

        self.conv0 = nn.Conv2d(2048, 1024, 1, 1, 0)
        self.conv1 = nn.Conv2d(512, 256, 1, 1, 0)
        self.conv2 = nn.Conv2d(512, 256, 1, 1, 0)
        self.conv3 = nn.Conv2d(256, 1, 1, 1, 0)
        self.conv4 = nn.Conv2d(128, 1, 1, 1, 0)

        self.bn0 = nn.BatchNorm2d(1024)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(1)

        self.relu = nn.ReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()
        #upsample function
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)

#        self.CA0 = Attention(64)
        self.CA1 = Attention(256)
        self.CA2 = Attention(512)
        self.CA3 = Attention(1024)
        self.CA4 = Attention(2048)

        self.FTM1 = FTM(256)
        self.FTM2 = FTM(64)

        if self.training:
            self.initialize_weights()

    def forward(self, x, x_depth):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x0 = self.resnet.maxpool(x)

        x_depth = self.resnet_depth.conv1(x_depth)
        x_depth = self.resnet_depth.bn1(x_depth)
        x_depth = self.resnet_depth.relu(x_depth)
        x0_depth = self.resnet_depth.maxpool(x_depth)

        x1 = self.resnet.layer1(x0)
        x2 = self.resnet.layer2(x1)
        x3_1 = self.resnet.layer3_1(x2)
        x4_1 = self.resnet.layer4_1(x3_1)
        x1_depth=self.resnet_depth.layer1(x0_depth)
        x2_depth=self.resnet_depth.layer2(x1_depth)
        x3_1_depth=self.resnet_depth.layer3_1(x2_depth)
        x4_1_depth = self.resnet_depth.layer4_1(x3_1_depth)

#        x0 = self.CA0(x0_depth, x0)
#        x0_depth = x0_depth + x0

#        x1_depth = self.resnet_depth.layer1(x0_depth)
        x1 = self.CA1(x1_depth, x1)
#        x1_depth = x1_depth +x1

#        x2_depth = self.resnet_depth.layer2(x1_depth)  # 512 x 32 x 32
        x2 = self.CA2(x2_depth, x2)
#        x2_depth = x2_depth + x2

#        x2_1 = x2
#        x3_1_depth = self.resnet_depth.layer3_1(x2_depth)  # 1024 x 16 x 16
        x3_1 = self.CA3(x3_1_depth, x3_1)
#        x3_1_depth = x3_1_depth + x3_1

#        x4_1_depth = self.resnet_depth.layer4_1(x3_1_depth)  # 2048 x 8 x 8
        x4_1 = self.CA4(x4_1_depth, x4_1)
#        x4_1_depth = x4_1_depth + x4_1

        x4_1 = self.relu(self.bn0(self.conv0(self.upsample2(x4_1))))
        s1 = x4_1 + x3_1
        s1 = self.FTM1(s1)
        x2_1 = self.relu(self.bn1(self.conv1(self.upsample2(x2))))
        s2 = x2_1 + x1
        s1 = self.relu(self.bn2(self.conv2(self.upsample2(s1))))

        s2 = s1 + s2
        s2 = self.FTM2(s2)
        s1 = self.relu(self.bn3(self.conv3(self.upsample4(s1))))
        s2 = self.relu(self.bn3(self.conv4(self.upsample2(s2))))

#        x4_1, x3_1, x2, x1 = self.Refine(x4_1, x3_1, x2, x1)
#        s4 = self.FAB_34(x4_1, x3_1)
#        s3 = self.FAB_23(x3_1, x2_1)
#        s2 = self.FAB_12(x2, x1)
#        s4 = self.FTM_4(s4)
#        s4 = self.relu(self.bn0(self.conv0(s4)))
#        print(s4.size())
#        print(s3.size())
#        s3 = s3 + s4.mul(self.sigmoid(s4))
#        s3 = self.FTM_3(s3)
#        s3 = self.relu(self.bn1(self.conv1(s3)))
#        s2 = s2 + s4.mul(self.sigmoid(s4))
#        s2 = self.FTM_2(s2)
#        s2 = self.FTM_1(s2)
#        s_0 = self.relu(self.bn2(self.conv2(s2)))
#        s_1 = self.relu(self.bn3(self.conv3(self.upsample4(s3))))
#        s_2 = self.relu(self.bn4(self.conv4(self.upsample8(s4))))
#        s_l = self.relu(self.bn2(self.conv2(s_l)))
#        s = self.relu(self.bn3(self.conv3(self.upsample2(s))))
#        s_l = s_l + s.mul(self.sigmoid(s))
#        s_l = self.relu(self.bn4(self.conv4(s_l)))

        return  s1, s2
     
    #initialize the weights
    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)

        all_params = {}
        for k, v in self.resnet_depth.state_dict().items():
            if k=='conv1.weight':
                all_params[k]=torch.nn.init.normal_(v, mean=0, std=1)
            elif k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet_depth.state_dict().keys())
        self.resnet_depth.load_state_dict(all_params)
"""""
def train_calibration(calibration_loader, model_rgb, model_depth, model_discriminator, model_estimator,
          optimizer_dis, optimizer_estimator, epoch,key_min):
    model_rgb.eval()
    model_depth.eval()
    model_discriminator.train()
    model_estimator.train()

    #for i in trange(calibration_loader.size):
    for i , pack in enumerate(tqdm(calibration_loader),start=1):
        iteration = i + epoch * len(calibration_loader)

        optimizer_dis.zero_grad()
        optimizer_estimator.zero_grad()

        # images, gts, depths, name = calibration_loader.load_data()
        images, gts, depths, name = pack
        images = Variable(images)
        gts = Variable(gts)
        depths = Variable(depths)
        cuda = torch.cuda.is_available()

        if cuda:
            images = images.cuda()
            gts = gts.cuda()
            depths = depths.cuda()
"""""
