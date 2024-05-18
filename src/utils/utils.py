import os
import sys

import yaml


def load_config(file_name="config.yaml"):
    with open(file_name, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["experiment_folder"] = os.path.join(
        config["log_dir"], config["experiment_name"]
    )
    config["train_logs_dir"] = os.path.join(config["experiment_folder"], "train_logs")
    config["dataset_file"] = os.path.join(
        config["experiment_folder"], "ds", "dataset.npz"
    )
    config["raw_dataset_folder"] = os.path.join(config["experiment_folder"], "raw")
    config["dataset_folder"] = os.path.join(config["experiment_folder"], "ds")
    return config


class RedirectStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
