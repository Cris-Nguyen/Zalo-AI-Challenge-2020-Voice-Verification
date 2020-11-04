import argparse
from src.utils import load_config, train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/ResNet50.yml', help='path to config file')
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)
