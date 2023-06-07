import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

import argparse

from src.utils.data_utils import load_and_split
from src.utils.load_params import load_params


def data_load(params):
    # dataset_url = params.data_load.dataset_url
    data_dir = Path(params.data_load.data_dir)
    train_file_path = Path(params.data_load.train_file_path)
    test_file_path = Path(params.data_load.test_file_path)
    split = params.data_load.split

    load_and_split(data_dir, train_file_path, test_file_path, split)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    params_path = args.config
    params = load_params(params_path)
    data_load(params)
