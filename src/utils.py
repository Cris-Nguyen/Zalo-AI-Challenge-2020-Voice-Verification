### Module imports ###
import os
import yaml
import librosa
from fastprogress import master_bar, progress_bar

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import SpeechDataset, mono_to_color
from src.models import ResNet


### Global Variables ###


### Class declarations ###


### Function declarations ###
def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def get_loader(config):
    train_dataset = SpeechDataset(config['data']['train'], config['audio_parameters'], config['melspectrogram_parameters'], config['img_size'])
    dev_dataset = SpeechDataset(config['data']['dev'], config['audio_parameters'], config['melspectrogram_parameters'], config['img_size'])
    train_loader = DataLoader(train_dataset, batch_size=config['train_parameters']['batch_size'], shuffle=config['train_parameters']['shuffle'])
    dev_loader = DataLoader(dev_dataset, batch_size=config['train_parameters']['batch_size'], shuffle=config['train_parameters']['shuffle'])
    return train_loader, dev_loader


def get_model(config):
    model_config = config['model']
    model_name = model_config['name']
    model_params = model_config['params']

    if 'resnet' in model_name:
        model = ResNet(  # type: ignore
            base_model_name=model_name,
            pretrained=model_params['pretrained'],
            num_classes=model_params['n_classes'])
        return model
    else:
        raise NotImplementedError


def get_criterion():
    criterion = nn.CrossEntropyLoss()
    return criterion


def get_optimizer(model, config):
    return optim.Adam(params=model.parameters(), **config['train_parameters']['optimizer'])


def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def train(config):
    # Get data, device, model, loss function nad optimizer for training
    train_loader, dev_loader = get_loader(config)
    device = get_device()
    model = get_model(config).to(device)
    criterion = get_criterion()
    optimizer = get_optimizer(model, config)
    performance = 0
    mb = master_bar(range(config['train_parameters']['epochs']))

    # Training
    for epoch in mb:
        print(f'Training epoch: {epoch + 1}')
        model.train()
        for inputs, labels in progress_bar(train_loader, parent=mb):
            print(inputs.shape)
            inputs = inputs.to(device)
            labels = labels.to(device)

            model.zero_grad()
            optimizer.zero_grad()

            predictions = model(inputs)
            predictions = predictions.to(device)

            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

        # Evaluate
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = datai
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predictions = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
            print(f'Accuracy: {correct / total}')

            if correct / total > performance:
                performance = round(correct / total, 3)
                if not os.path.exists(cf.models_path):
                    os.makedirs(cf.models_path)
                torch.save(model.state_dict(), f'{config["saved_model"]}/{config["model"]["name"]}_{performance}.pkl')
            print(f'Best accuracy: {performance * 100}')
            print('*' * 10)


def load_model(model_path, config):
    model = get_model(config)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def inference(model, filepath, config):
    # Preprocess audio from raw wav 
    y, sr = librosa.load(filepath, config['audio_parameters']['sampling_rate'])
    len_y = len(y)
    desired_sample = self.audio_parameters['desired_sample']
    if len_y < desired_sample:
        new_y = np.zeros(desired_sample, dtype=y.dtype)
        start = np.random.randint(desired_sample - len_y)
        new_y[start:start + len_y] = y
        y = new_y
    else:
        start = np.random.randint(len_y - desired_sample)
        y = y[start:start + desired_sample]

        melspec = librosa.feature.melspectrogram(y, sr=audio_parameters['sampling_rate'], **self.
melspectrogram_parameters)
        melspec = librosa.power_to_db(melspec).astype(np.float32)
        image = mono_to_color(melspec)
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        # image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)
        image = torch.from_numpy(image)

    # Feed features to model
    output = model(image)
    _, predictions = torch.max(outputs.data, dim=1)
    return predictions


if __name__ == '__main__':
    config = {'data': {'train': '../Train-Test-Data/train.csv', 'dev':'../Train-Test-Data/dev.csv'},
              'audio_parameters': {'sampling_rate': 16000, 'desired_sample': 16000},
              'melspectrogram_parameters': {'n_mels': 128, 'fmin': 20, 'fmax': 16000},
              'img_size': 224,
              'model': {'name': 'resnet50', 'params': {'pretrained': False, 'n_classes': 400}},
              'train_parameters': {'batch_size': 64, 'shuffle': True, 'optimizer': {'lr': 1e-6, 'amsgrad': False}}
             }
    config = load_config('../config/ResNet50.yml')
    print(config)
    device = get_device()
    train_loader, dev_loader = get_loader(config)
    model = get_model(config)
    optimizer = get_optimizer(model, config)
