### Module imports ###
import os
import yaml
import numpy as np
import pandas as pd
import librosa
import cv2
from fastprogress import master_bar, progress_bar

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import SpeechDataset, mono_to_color
from src.models import ResNet, MobileNet, MNASNet, ConvAngularPen


### Global Variables ###


### Class declarations ###


### Function declarations ###
def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def get_loader(csv_path, config):
    dataset = SpeechDataset(csv_path, config['audio_parameters'], config['melspectrogram_parameters'], config['img_size'])
    loader = DataLoader(dataset, batch_size=config['train_parameters']['batch_size'], shuffle=config['train_parameters']['shuffle'])
    return loader


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


def get_criterion():
    criterion = nn.CrossEntropyLoss()
    return criterion


def get_optimizer(model, config):
    return optim.Adam(params=model.parameters(), **config['train_parameters']['optimizer'])


def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def train(config):
    # Get data, device, model, loss function nad optimizer for training
    train_loader = get_loader(config['data']['train'], config)
    dev_loader = get_loader(config['data']['dev'], config)
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
                inputs, labels = data
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


def train_am(config):
    # Get data, device, model, loss function nad optimizer for training
    train_loader = get_loader(config['data']['train'], config)
    device = get_device()
    model = get_model(config).to(device)
    optimizer = get_optimizer(model, config)
    total_step = len(train_loader)
    mb = master_bar(range(config['train_parameters']['epochs']))

    # Training
    for epoch in mb:
        print(f'Training epoch: {epoch + 1}')
        model.train()
        for i, (inputs, labels) in enumerate(progress_bar(train_loader, parent=mb)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # model.zero_grad()
            loss = model(inputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch}/{config["train_parameters"]["epochs"]}], Step [{i + 1}/{total_step}], Loss: {loss.item()}')
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), f'{config["saved_model"]}/{config["model"]["name"]}_{round(loss.item(), 5)}.pkl')
            if (epoch + 1) % 20 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / 4


def load_model(model_path, config):
    model = get_model(config)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def inference(model, filepath, config):
    # Preprocess audio from raw wav 
    y, sr = librosa.load(filepath, config['audio_parameters']['sampling_rate'])
    len_y = len(y)
    desired_sample = config['audio_parameters']['desired_sample']
    if len_y < desired_sample:
        new_y = np.zeros(desired_sample, dtype=y.dtype)
        start = np.random.randint(desired_sample - len_y)
        new_y[start:start + len_y] = y
        y = new_y
    else:
        start = np.random.randint(len_y - desired_sample)
        y = y[start:start + desired_sample]

    melspec = librosa.feature.melspectrogram(y, sr=config['audio_parameters']['sampling_rate'], **config['melspectrogram_parameters'])
    melspec = librosa.power_to_db(melspec).astype(np.float32)
    image = mono_to_color(melspec)
    height, width, _ = image.shape
    image = cv2.resize(image, (int(width * config['img_size'] / height), config['img_size']))
    # image = np.moveaxis(image, 2, 0)
    image = (image / 255.0).astype(np.float32)
    image = torch.from_numpy(image).view(1, image.shape[2], image.shape[0], image.shape[1])

    # Feed features to model
    output = model(image)
    return output
    _, predictions = torch.max(output.data, dim=1)
    return predictions


def make_submission(model, config, csv_path, output):
    orig_folder = 'Train-Test-Data/public-test/'
    df_data = pd.read_csv(csv_path)
    audio_1, audio_2 = df_data['audio_1'], df_data['audio_2']
    label = []
    for i in range(len(audio_1)):
        filepath1 = os.path.join(orig_folder, audio_1[i])
        filepath2 = os.path.join(orig_folder, audio_2[i])
        output1 = inference(model, filepath1, config).detach().numpy()
        output2 = inference(model, filepath2, config).detach().numpy()
        dis = np.sqrt(np.sum(np.power((output1 - output2), 2)))
        print(f'{audio_1[i]} {audio_2[i]} {dis}\n')
        if dis < 50:
            label.append(1)
        else:
            label.append(0)

    df_out = pd.DataFrame(columns=['audio_1', 'audio_2', 'label'])
    df_out['audio_1'] = audio_1
    df_out['audio_2'] = audio_2
    df_out['label'] = label
    df_out.to_csv(output, index=False)


if __name__ == '__main__':
    config = {'data': {'train': '../Train-Test-Data/train.csv', 'dev':'../Train-Test-Data/dev.csv'},
              'audio_parameters': {'sampling_rate': 16000, 'desired_sample': 16000},
              'melspectrogram_parameters': {'n_mels': 128, 'fmin': 20, 'fmax': 16000},
              'img_size': 224,
              'model': {'name': 'resnet50', 'params': {'pretrained': False, 'n_classes': 400}},
              'train_parameters': {'batch_size': 64, 'shuffle': True, 'optimizer': {'lr': 1e-6, 'amsgrad': False}}
             }
    config = load_config('../config/ResNet50.yml')
    model = load_model('../saved_model/resnet50_acc_0.785.pkl', config)
    print(model)
    device = get_device()
    train_loader, dev_loader = get_loader(config)
    model = get_model(config)
    optimizer = get_optimizer(model, config)
