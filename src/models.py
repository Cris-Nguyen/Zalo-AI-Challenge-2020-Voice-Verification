### Module imports ###
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary


### Global Variables ###


### Class declarations ###
class ResNet(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False,
                 num_classes=400):
        super().__init__()
        base_model = models.__getattribute__(base_model_name)(pretrained=pretrained)
        layers = list(base_model.children())[:-2]
        layers.append(nn.AdaptiveMaxPool2d(1))
        self.encoder = nn.Sequential(*layers)

        in_features = base_model.fc.in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        multiclass_proba = F.softmax(x, dim=1)
        return multiclass_proba
        # multilabel_proba = F.sigmoid(x)
        # return {
        #     "logits": x,
        #     "multiclass_proba": multiclass_proba,
        #     "multilabel_proba": multilabel_proba
        # }


### Function declarations ###
def get_model(config: dict):
    model_config = config["model"]
    model_name = model_config["name"]
    model_params = model_config["params"]

    if "resnet" in model_name:
        model = ResNet(  # type: ignore
            base_model_name=model_name,
            pretrained=model_params["pretrained"],
            num_classes=model_params["n_classes"])
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
        'model': {'name': 'resnet50', 'params': {'pretrained': False, 'n_classes': 400}}
    }
    model = get_model(config).cuda()
    print(count_parameters(model))
    print_size_of_model(model)
    summary(model, (3, 224, 164))
