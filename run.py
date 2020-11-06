import os
import argparse
from src.utils import load_config, train, train_am, load_model, inference, make_submission


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/ResNet50.yml', help='path to config file')
    args = parser.parse_args()

    config = load_config(args.config)
    train_am(config)
    # model = load_model('saved_model/resnet50_acc_0.785.pkl', config)
    # make_submission(model, config, 'Train-Test-Data/public-test.csv', 'submission.csv')
    # orig_folder = 'Train-Test-Data/dataset/439-F-38'
    # files = os.listdir(orig_folder)
    # for fn in files:
    #     output = inference(model, os.path.join(orig_folder, fn), config)
    #     print(output.item())
