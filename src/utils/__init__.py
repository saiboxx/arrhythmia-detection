import os
import pickle
from datetime import datetime
from typing import Optional

import torch
import yaml
from torch import tensor
from torch.utils.tensorboard import SummaryWriter


class TrackingAgent(object):
    def __init__(self, batch_size: int, num_samples: int):
        self.batch_size = batch_size
        self.num_samples = num_samples

        self.losses = []
        self.correct_class = 0

    def add_loss(self, loss: tensor):
        self.losses.append(float(loss))

    def add_correct_class(self, y_hat: tensor, y: tensor):
        with torch.no_grad():
            self.correct_class += (torch.argmax(y_hat, dim=1) == y).float().sum()

    def get_loss(self) -> float:
        return sum(self.losses) / len(self.losses)

    def get_accuracy(self) -> float:
        return self.correct_class / self.num_samples

    def reset(self):
        self.losses = []
        self.correct_class = 0


class SummaryAgent(object):
    """
    Logs metrics to tensorboard files
    """

    def __init__(self, directory: str, model_name: str, cfg: Optional[dict] = None):
        """
        Initializes a summary object.
        :param directory: Saving directory of dirs
        :param model_name: Subfolder for the logs
        :param cfg: Optional dictionary with parameters to be saved.
        """
        self.directory = os.path.join(directory,
                                      model_name,
                                      datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.model_dir = os.path.join('models',
                                      model_name,
                                      datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.directory, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.directory)
        self.episode = 1

        if cfg is not None:
            params = {
                'NETWORK': model_name,
                'EPOCHS': int(cfg['EPOCHS']),
                'BATCH_SIZE': int(cfg['BATCH_SIZE']),
                'HIDDEN_SIZE': int(cfg['HIDDEN_SIZE']),
                'NUM_LAYERS': int(cfg['NUM_LAYERS']),
                'DROPOUT': int(cfg['DROPOUT']),
                'LEARNING_RATE': cfg['LR']
            }
            self.writer.add_hparams(hparam_dict=params, metric_dict={})

    def save_model(self, model: torch.nn.Module):
        path = os.path.join(self.model_dir, str(self.episode) + '.pt')
        torch.save(model.state_dict(), path)

    def add_scalar(self, tag: str, value):
        """
        Add a scalar to the summary.
        :param tag: Tag of scalar
        :param value: Value of scalar
        """
        step = self.episode

        self.writer.add_scalar(tag, value, step)

    def adv_episode(self):
        """
        Increase episode counter
        """
        self.episode += 1

    def close(self):
        """
        Flush the cached metrics and close writer.
        """
        self.writer.flush()
        self.writer.close()

    def flush(self):
        """
        Flush the cached metrics
        """
        self.writer.flush()


class Timer(object):
    """
    Utility class for a simple tracking of wall time.
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        print(self.name + " ...")
        self.start = datetime.now()

    def __exit__(self, type, value, traceback):
        self.stop = datetime.now()
        delta = self.stop - self.start
        seconds = delta.seconds
        minutes, seconds_of_minute = divmod(seconds, 60)
        hours, minutes_of_hour = divmod(minutes, 60)
        print(self.name + " took {:02}:{:02}:{:02}".format(int(hours), int(minutes_of_hour), int(seconds_of_minute)))


def load_config() -> dict:
    """
    Loads the config.yml file to memory and returns it as dictionary.
    :return: Dictionary containing the config.
    """
    with open('config.yml', 'r') as ymlfile:
        return yaml.load(ymlfile, Loader=yaml.FullLoader)


def save_pickle(file: object, filepath: str):
    """
    Saves a generic object to a binary file.
    :param file: File to be saved
    :param filepath: Saving destination
    """
    with open(filepath, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath: str) -> object:
    """
    Loads a pickle file to memory.
    :param filepath: Path pointing to file
    """
    with open(filepath, 'rb') as handle:
        return pickle.load(handle)
