import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

import argparse

from src.utils.load_params import load_params
from src.utils.train_utils import train_model


def train_and_save_model(params):
    train_file_path = Path(params.data_load.train_file_path)
    epochs = params.train.epochs
    batch_size = params.train.batch_size
    validation_split = params.train.validation_split
    model_path = params.train.model_path

    train_model(train_file_path, epochs, batch_size, validation_split, model_path)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    params_path = args.config
    params = load_params(params_path)
    train_and_save_model(params)
