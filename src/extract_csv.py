### Module imports ###
import os
import wave
import contextlib
import pandas as pd
from sklearn.model_selection import train_test_split as split


### Global Variables ###
orig_folder = 'Train-Test-Data/dataset/'


### Class declarations ###


### Function declarations ###
def get_duration(wav_path):
    with contextlib.closing(wave.open(wav_path,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        print(rate)
        duration = frames / float(rate)
    return duration


def extract_csv(path, label, output):
    df = pd.DataFrame(columns=['path', 'label'])
    df['path'] = path
    df['label'] = label
    df.to_csv(output, index=False)


def extract_data(data_dir, train_csv, dev_csv, class_txt):
    class_dict = {}
    class_idx = 0
    audio_train, label_train, audio_dev, label_dev = [], [], [], []
    classes = os.listdir(data_dir)

    for idx in classes:
        class_dict[idx] = class_idx
        class_path = os.path.join(data_dir, idx)
        files = os.listdir(class_path)
        path = [os.path.abspath(os.path.join(class_path, fn)) for fn in files]
        X_train, X_dev, y_train, y_dev = split(path, [class_idx] * (len(files)), test_size=0.2)
        audio_train += X_train
        label_train += y_train
        audio_dev += X_dev
        label_dev += y_dev
        class_idx += 1

    extract_csv(audio_train, label_train, train_csv)
    extract_csv(audio_dev, label_dev, dev_csv)

    f = open(class_txt, 'w')
    for idx in class_dict:
        f.write(f'{idx}\t{class_dict[idx]}\n')
    f.close()


if __name__ == '__main__':
    extract_data('../Train-Test-Data/dataset', '../Train-Test-Data/train.csv', '../Train-Test-Data/dev.csv', '../Train-Test-Data/class_dict.txt')
