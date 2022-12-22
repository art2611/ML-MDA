import torch
import torch.nn as nn
from torchvision.models import resnet18


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Flatten(nn.Module):
    def forward(self, x):
        return(x.view(x.size(0), x.size(1)))

class visible_module(nn.Module):
    def __init__(self, fusion_layer=4, pool_dim = 512):
        super(visible_module, self).__init__()

        self.visible = resnet18(pretrained=True)

        self.fusion_layer = fusion_layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)

        layer0 = [self.visible.conv1, self.visible.bn1, self.visible.relu, self.visible.maxpool]
        self.layer_dict = {"layer0": nn.Sequential(*layer0), "layer1": self.visible.layer1,
                           "layer2": self.visible.layer2, "layer3": self.visible.layer3, "layer4": self.visible.layer4,
                           "layer5": self.avgpool, "layer6": Flatten(), "layer7": self.bottleneck}

    def forward(self, x, with_features = False):
        for i in range(0, self.fusion_layer):
            if i == 5:
                backbone_feat = x
                x_pool = self.layer_dict["layer5"](x)
                x_pool = self.layer_dict["layer6"](x_pool)
                feat = self.layer_dict["layer7"](x_pool)
                if with_features:
                    return [x_pool, feat, backbone_feat]
                return [x_pool, feat]
            if i < 5:
                x = self.layer_dict["layer" + str(i)](x)

        return x


    def count_params(self):
        s = 0
        for layer in self.layer_dict["layer0"]:
            s += count_parameters(layer)
        for i in range(1, self.fusion_layer):
            s += count_parameters(self.layer_dict["layer" + str(i)])
        return s


class thermal_module(nn.Module):
    def __init__(self, fusion_layer=4, pool_dim = 512):
        super(thermal_module, self).__init__()

        self.thermal = resnet18(pretrained=True)

        self.fusion_layer = fusion_layer

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)

        layer0 = [self.thermal.conv1, self.thermal.bn1, self.thermal.relu, self.thermal.maxpool]
        self.layer_dict = {"layer0": nn.Sequential(*layer0), "layer1": self.thermal.layer1,
                           "layer2": self.thermal.layer2, "layer3": self.thermal.layer3, "layer4": self.thermal.layer4,
                           "layer5": self.avgpool, "layer6": Flatten(), "layer7": self.bottleneck}

    def forward(self, x, with_features = False):
        for i in range(0, self.fusion_layer):
            if i == 5:
                backbone_feat = x
                x_pool = self.layer_dict["layer5"](x)
                x_pool = self.layer_dict["layer6"](x_pool)
                feat = self.layer_dict["layer7"](x_pool)
                if with_features :
                    return[x_pool, feat, backbone_feat]
                return [x_pool, feat]
            if i < 5:
                x = self.layer_dict["layer" + str(i)](x)
        return x

    def count_params(self):
        s = 0
        for layer in self.layer_dict["layer0"]:
            s += count_parameters(layer)
        for i in range(1, self.fusion_layer):
            s += count_parameters(self.layer_dict["layer" + str(i)])
        return s


class shared_resnet(nn.Module):
    def __init__(self, fusion_layer=4, pool_dim = 512):
        super(shared_resnet, self).__init__()

        self.fusion_layer = fusion_layer

        model_base = resnet18(pretrained=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.base = model_base

        layer0 = [self.base.conv1, self.base.bn1, self.base.relu, self.base.maxpool]
        self.layer_dict = {"layer0": nn.Sequential(*layer0), "layer1": self.base.layer1,
                           "layer2": self.base.layer2, "layer3": self.base.layer3, "layer4": self.base.layer4,
                           "layer5": self.avgpool, "layer6": Flatten(), "layer7": self.bottleneck}

    def forward(self, x):

        for i in range(self.fusion_layer, 6):
            if i < 5 :
                x = self.layer_dict["layer" + str(i)](x)
            else :
                x_pool = self.layer_dict["layer5"](x)
                x_pool = self.layer_dict["layer6"](x_pool)
                feat = self.layer_dict["layer7"](x_pool)
                return [x_pool, feat]

    def count_params(self):
        s = 0
        for i in range(self.fusion_layer, 5):
            s += count_parameters(self.layer_dict["layer" + str(i)])
        return s

class Global_network(nn.Module):
    def __init__(self, class_num, fusion_layer=4, model="", attention_needed=False):
        super(Global_network, self).__init__()

        pool_dim = 512
        self.pool_dim = pool_dim

        self.visible_module = visible_module(fusion_layer=fusion_layer, pool_dim = pool_dim)
        self.thermal_module = thermal_module(fusion_layer=fusion_layer, pool_dim = pool_dim)

        self.model = model

        nb_modalities = 2

        self.fusion_layer = fusion_layer

        self.shared_resnet = shared_resnet(fusion_layer=fusion_layer, pool_dim = pool_dim)

        if model == "concatenation" :
            pool_dim = 2*pool_dim

        self.fc = nn.Linear(pool_dim, class_num, bias=False)
        self.l2norm = Normalize(2)

        self.nb_modalities = nb_modalities

    def forward(self, X, model="concatenation", modality="BtoB"):
        if model == "unimodal":
            if modality == "VtoV" :
                x_pool, feat = self.visible_module(X[0])
            elif modality == "TtoT":
                x_pool, feat= self.thermal_module(X[1])
        else :
            X[0] = self.visible_module(X[0]) # X[0] = (X_pool, feat)
            X[1] = self.thermal_module(X[1]) # X[1] = (X_pool, feat)

            x_pool, feat = torch.cat((X[0][0], X[1][0]), 1), torch.cat((X[0][1], X[1][1]), 1)

        if self.training:
            return x_pool, self.fc(feat)
        else:
            return self.l2norm(x_pool), self.l2norm(feat)

    def count_params(self, model):
        global s
        # Not adapted to GAN_mmodal
        if model != "unimodal":

            s = self.visible_module.count_params() + self.thermal_module.count_params()
            s += self.shared_resnet.count_params()

            s += sum(p.numel() for p in self.fc.parameters() if p.requires_grad)

            # In this case : Unimodal
        elif model == "unimodal":

            s = self.visible_module.count_params() + self.shared_resnet.count_params() + sum(
                p.numel() for p in self.fc.parameters() if p.requires_grad)
        return s
