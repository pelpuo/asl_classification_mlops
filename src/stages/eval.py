import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

import argparse
import json
from src.utils.eval_utils import get_metrics
from src.utils.load_params import load_params


def evaluate(params):
    test_file_path = Path(params.data_load.test_file_path)
    metrics_file = Path(params.evaluate.metrics_file)
    model_path = params.train.model_path

    metrics = get_metrics(test_file_path, model_path)

    Path(params.evaluate.metrics_file).parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj=metrics, fp=open(metrics_file, "w"), indent=4)

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    params_path = args.config
    params = load_params(params_path)
    evaluate(params)
