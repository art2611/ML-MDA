import torch
import torch.nn as nn
from torchvision.models import resnet18 as resnet50

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Flatten(nn.Module):
    def forward(self, x):
        return(x.view(x.size(0), x.size(1)))

class visible_module(nn.Module):
    def __init__(self, fusion_layer=4, pool_dim = 512):
        super(visible_module, self).__init__()

        self.visible = resnet50(pretrained=True)

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

        self.thermal = resnet50(pretrained=True)

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

class Global_CIL(nn.Module):
    def __init__(self, class_num, fusion_layer=4, fuse = "", fusion="", attention_needed=False):
        super(Global_CIL, self).__init__()


        pool_dim = 512 # Resnet18 pool dim
        # pool_dim = 2048 # Resnet50 pool dim
        self.pool_dim = pool_dim

        self.visible_module = visible_module(fusion_layer=fusion_layer, pool_dim = pool_dim)
        self.thermal_module = thermal_module(fusion_layer=fusion_layer, pool_dim = pool_dim)

        self.fusion = fusion

        nb_modalities = 2

        self.fusion_layer = fusion_layer

        if fuse == "concatenation" : # Added 18/07/2022
            pool_dim = 2*pool_dim

        self.fc = nn.Linear(pool_dim, class_num, bias=False)
        self.l2norm = Normalize(2)

        self.nb_modalities = nb_modalities

    def forward(self, X, modal=0, fuse="cat_channel", modality="BtoB", with_features=False, get_after_MAN=False, get_before_MAN=False):
        x, z, att = 0, 0, 0
        if fuse == "none":
            if modality == "VtoV" or modality == "BtoV":
                if with_features :
                    x_pool, feat, backbone_feat = self.visible_module(X[0], with_features)
                else :
                    x_pool, feat = self.visible_module(X[0], with_features)
            elif modality == "TtoT" or modality == "BtoT":
                x_pool, feat= self.thermal_module(X[1])
        else :
            X[0] = self.visible_module(X[0], with_features) # X[0] = (X_pool, feat, if with_features backbone_feat here)
            X[1] = self.thermal_module(X[1], with_features) # X[1] = (X_pool, feat, same)
            if with_features:
                backbone_feat = torch.cat((X[0][2], X[1][2]), 1)

            if fuse == "concatenation" or fuse == "fc_fuse":
                x_pool, feat = torch.cat((X[0][0], X[1][0]), 1), torch.cat((X[0][1], X[1][1]), 1)

            elif fuse == "sum":
                x_pool, feat = X[0][0], X[0][1]
                for i in range(2 - 1):  # Add all the upcoming modalities
                    x_pool, feat = x_pool.add(X[i + 1][0]), feat.add(X[i + 1][1])

        if with_features:
            return x_pool, self.fc(feat), backbone_feat
        if self.training:
            return x_pool, self.fc(feat)
        else:
            return self.l2norm(x_pool), self.l2norm(feat), feat, att