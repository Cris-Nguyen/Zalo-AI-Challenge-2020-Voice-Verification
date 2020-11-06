### Module imports ###
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary

from src.loss_functions import AngularPenaltySMLoss


### Global Variables ###


### Class declarations ###
class ResNet(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False,
                 num_classes=400, embed=False):
        super().__init__()
        base_model = models.__getattribute__(base_model_name)(pretrained=pretrained)
        layers = list(base_model.children())[:-2]
        layers.append(nn.AdaptiveMaxPool2d(1))
        self.encoder = nn.Sequential(*layers)

        in_features = base_model.fc.in_features
        self.fc1 = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(p=0.2))
        self.fc2 = nn.Linear(512, num_classes)
        self.embed = embed

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        if self.embed:
            return x
        x = self.fc2(x)
        return x
        # multiclass_proba = F.softmax(x, dim=1)
        # return multiclass_proba
        # multilabel_proba = F.sigmoid(x)
        # return {
        #     "logits": x,
        #     "multiclass_proba": multiclass_proba,
        #     "multilabel_proba": multilabel_proba
        # }


class MobileNet(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False, num_classes=400, embed=False):
        super().__init__()
        base_model = models.__getattribute__(base_model_name)(pretrained=pretrained)
        layers = list(base_model.children())[:-1]
        layers.append(nn.AdaptiveMaxPool2d(1))
        self.encoder = nn.Sequential(*layers)
        self.fc1 = nn.Sequential(nn.Linear(1280, 512), nn.ReLU(), nn.Dropout(0.2))
        self.fc2 = nn.Linear(512, num_classes)
        self.embed = embed

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        if self.embed:
            return x
        x = self.fc2(x)
        return x


class MNASNet(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False, num_classes=400, embed=False):
        super().__init__()
        base_model = models.__getattribute__(base_model_name)(pretrained=pretrained)
        layers = list(base_model.children())[:-1]
        layers.append(nn.AdaptiveMaxPool2d(1))
        self.encoder = nn.Sequential(*layers)
        self.fc1 = nn.Sequential(nn.Linear(1280, 512), nn.ReLU(), nn.Dropout(0.2))
        self.fc2 = nn.Sequential(nn.Linear(512, num_classes))
        self.embed = embed

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        if self.embed:
            return x
        x = self.fc2(x)
        return x


class ConvAngularPen(nn.Module):
    def __init__(self, conv_model_name, pretrained=False, num_classes=400, loss_type='arcface', embed=False):
        super(ConvAngularPen, self).__init__()
        if conv_model_name == 'resnet50':
            convmodel = ResNet(base_model_name=conv_model_name, pretrained=pretrained, num_classes=num_classes, embed=True)
        elif conv_model_name == 'mobilenet_v2':
            convmodel = MobileNet(base_model_name=conv_model_name, pretrained=pretrained, num_classes=num_classes, embed=True)
        elif conv_model_name == 'mnasnet1_0':
            convmodel = MNASNet(base_model_name=conv_model_name, pretrained=pretrained, num_classes=num_classes, embed=True)

        self.convlayers = convmodel
        self.adms_loss = AngularPenaltySMLoss(512, num_classes, loss_type=loss_type)
        self.embed = embed

    def forward(self, x, labels=None):
        x = self.convlayers(x)
        if self.embed:
            return x
        L = self.adms_loss(x, labels)
        return L


### Function declarations ###
def get_model(config: dict):
    model_config = config['model']
    model_name = model_config['name']
    model_params = model_config['params']

    if 'angpen' in model_name:
        conv_model_name = model_name.split('-')[1]
        loss_type = model_name.split('-')[2]
        model = ConvAngularPen(
                conv_model_name,
                pretrained=model_params['pretrained'],
                num_classes=model_params['n_classes'],
                loss_type=loss_type,
                embed=model_params['embed'])
        return model
    elif 'resnet' in model_name:
        model = ResNet(  # type: ignore
                base_model_name=model_name,
                pretrained=model_params['pretrained'],
                num_classes=model_params['n_classes'],
                embed=model_params['embed'])
        return model
    elif 'mobilenet' in model_name:
        model = MobileNet(
                base_model_name=model_name,
                pretrained=model_params['pretrained'],
                num_classes=model_params['n_classes'],
                embed=model_params['embed'])
        return model
    elif 'mnasnet' in model_name:
        model = MNASNet(
                base_model_name=model_name,
                pretrained=model_params['pretrained'],
                num_classes=model_params['n_classes'],
                embed=model_params['embed'])
        return model
    else:
        raise NotImplementedError


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_size_of_model(model):
    """ Compute size of model """
    torch.save(model.state_dict(), 'temp.p')
    print('Size (MB):', os.path.getsize('temp.p')/1e6)
    os.remove('temp.p')


if __name__ == '__main__':
    config = {
        'model': {'name': 'mobilenet_v2', 'params': {'pretrained': False, 'n_classes': 400, 'embed': False}}
    }
    model = get_model(config).cuda()
    print(count_parameters(model))
    print_size_of_model(model)
    summary(model, (3, 224, 164))
