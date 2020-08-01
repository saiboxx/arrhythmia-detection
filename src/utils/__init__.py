import os
import pickle
import warnings
from datetime import datetime
from time import time
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import nvidia_smi
import psutil
import torch
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import tensor
from torch.utils.tensorboard import SummaryWriter


class TrackingAgent(object):
    """
    An agent that keeps track of metrics and other trackable attributes during training.
    """

    def __init__(self):
        """
        Initialize a TrackingAgent instance.
        """
        self.time = None
        self.epoch_time = None
        self.cpu_usage = None
        self.gpu_usage = None

        nvidia_smi.nvmlInit()
        self.gpu_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

        # Summary stats per epoch
        self.stats = {
            'train_loss': [],
            'test_loss': [],
            'train_acc': [],
            'test_acc': [],
            'train_prec': [],
            'test_prec': [],
            'train_rec': [],
            'test_rec': [],
            'train_f1': [],
            'test_f1': [],
            'epoch_time': [],
            'cpu': [],
            'gpu': []
        }

        # Intermediate cache during epochs
        self.train_losses = []
        self.test_losses = []
        self.train_truth = []
        self.train_pred = []
        self.test_truth = []
        self.test_pred = []

        self.epoch_time = None
        self.cpu_usage = None
        self.gpu_usage = None

        warnings.filterwarnings('always')

    def add_train_loss(self, loss: tensor):
        """
        Adds batched training loss to intermediate loss list.
        :param loss: Batched training loss
        """
        self.train_losses.append(float(loss))

    def add_test_loss(self, loss: tensor):
        """
        Adds batched test loss to intermediate loss list.
        :param loss: Batched training loss
        """
        self.test_losses.append(float(loss))

    def add_train_prediction(self, y_hat: tensor, y: tensor):
        """
        Takes a c-dimensional scoring output from the model, where c is the
        number of classes and a truth label vector. Observation are sampled
         from the training set. Prediction and truth are saved in intermediate
         lists for further processing.
        :param y_hat: Tensor with shape (num_samples, num_classes)
        :param y: Truth label vector with shape (num_samples)
        """
        with torch.no_grad():
            self.train_truth.extend(y.tolist())
            self.train_pred.extend((torch.argmax(y_hat, dim=1).tolist()))

    def add_test_prediction(self, y_hat: tensor, y: tensor):
        """
        Takes a c-dimensional scoring output from the model, where c is the
        number of classes and a truth label vector. Observation are sampled
         from the test set. Prediction and truth are saved in intermediate
         lists for further processing.
        :param y_hat: Tensor with shape (num_samples, num_classes)
        :param y: Truth label vector with shape (num_samples)
        """
        with torch.no_grad():
            self.test_truth.extend(y.tolist())
            self.test_pred.extend((torch.argmax(y_hat, dim=1).tolist()))

    def get_train_loss(self) -> float:
        """
        Convert intermediate list of train losses to global loss for an epoch.
        :return Epoch loss
        """
        loss = sum(self.train_losses) / len(self.train_losses)
        self.stats['train_loss'].append(loss)
        return loss

    def get_test_loss(self) -> float:
        """
        Convert intermediate list of test losses to global loss for an validation epoch.
        :return Validation epoch loss
        """
        loss = sum(self.test_losses) / len(self.test_losses)
        self.stats['test_loss'].append(loss)
        return loss

    def get_train_metrics(self) -> Tuple:
        """
        Convert intermediate batched results to epoch metrics.
        Results are also added to summary dictionary.
        :return: Tuple consisting of accuracy, precision, recall, F1-score.
        """
        acc = accuracy_score(self.train_truth, self.train_pred)
        prec, rec, fscore, _ = precision_recall_fscore_support(self.train_truth,
                                                               self.train_pred,
                                                               warn_for=tuple(),
                                                               average='macro')

        self.stats['train_acc'].append(acc)
        self.stats['train_prec'].append(prec)
        self.stats['train_rec'].append(rec)
        self.stats['train_f1'].append(fscore)
        return acc, prec, rec, fscore

    def get_test_metrics(self) -> Tuple:
        """
        Convert intermediate batched results to epoch metrics.
        Results are also added to summary dictionary.
        :return: Tuple consisting of accuracy, precision, recall, F1-score.
        """
        acc = accuracy_score(self.test_truth, self.test_pred)
        prec, rec, fscore, _ = precision_recall_fscore_support(self.test_truth,
                                                               self.test_pred,
                                                               warn_for=tuple(),
                                                               average='macro')
        self.stats['test_acc'].append(acc)
        self.stats['test_prec'].append(prec)
        self.stats['test_rec'].append(rec)
        self.stats['test_f1'].append(fscore)
        return acc, prec, rec, fscore

    def get_plots(self, show: Optional[bool] = False):
        """
        Plot the metrics saved in the summary dictionary. Plots are also saved to
        disc and a summary is printed to console.
        :param show: Bool whether to show graphs or not
        """
        x_axis = [x for x in range(len(self.stats['train_loss']))]

        fig = plt.figure(1, figsize=(6, 3))
        ax1 = plt.subplot(1, 1, 1)
        ln1 = ax1.plot(x_axis, self.stats['train_loss'], label='Train Loss')
        ln2 = ax1.plot(x_axis, self.stats['test_loss'], label='Test Loss')
        ax1.set_ylim([0, 1.2])
        ax2 = ax1.twinx()
        ln3 = ax2.plot(x_axis, self.stats['train_f1'], 'g', label='Train F1-Score')
        ln4 = ax2.plot(x_axis, self.stats['test_f1'], 'r', label='Test F1-Score')
        ax2.set_ylim([0, 1])
        lns = ln1 + ln2 + ln3 + ln4
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)

        fig.tight_layout()

        os.makedirs('plots', exist_ok=True)
        fig.savefig('plots/results_loss.pdf', bbox_inches='tight')

        fig = plt.figure(2, figsize=(10, 10))

        ax1 = plt.subplot(4, 1, 1)
        ln1 = ax1.plot(x_axis, self.stats['train_loss'], label='Train Loss')
        ln2 = ax1.plot(x_axis, self.stats['test_loss'], label='Test Loss')
        ax2 = ax1.twinx()
        ln3 = ax2.plot(x_axis, self.stats['train_acc'], 'g', label='Train Accuracy')
        ln4 = ax2.plot(x_axis, self.stats['test_acc'], 'r', label='Test Accuracy')
        lns = ln1 + ln2 + ln3 + ln4
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='center right')
        plt.title('Loss & Accuracy')

        ax3 = plt.subplot(4, 1, 2)
        ax3.plot(x_axis, self.stats['train_prec'], label='Train Precision')
        ax3.plot(x_axis, self.stats['test_prec'], label='Test Precision')
        ax3.legend(loc='lower right')
        plt.title('Precision')

        ax4 = plt.subplot(4, 1, 3)
        ax4.plot(x_axis, self.stats['train_rec'], label='Train Recall')
        ax4.plot(x_axis, self.stats['test_rec'], label='Test Recall')
        ax4.legend(loc='lower right')
        plt.title('Recall')

        ax5 = plt.subplot(4, 1, 4)
        ax5.plot(x_axis, self.stats['train_f1'], label='Train F1-Score')
        ax5.plot(x_axis, self.stats['test_f1'], label='Test F1-Score')
        ax5.legend(loc='lower right')
        plt.title('F1-Score')

        fig.tight_layout()
        fig.savefig('plots/results.pdf', bbox_inches='tight')

        save_pickle(self.stats, 'plots/stats.pkl')

        best_round = self.stats['test_f1'].index(max(self.stats['test_f1']))

        print(('Best Epoch: {0}\n' +
               'Train Loss: {1:.3f}\n' +
               'Test Loss: {2:.3f}\n' +
               'Train Accuracy: {3:.2f}\n' +
               'Test Accuracy: {4:.2f}\n' +
               'Train Precision: {5:.2f}\n' +
               'Test Precision: {6:.2f}\n' +
               'Train Recall: {7:.2f}\n' +
               'Test Recall: {8:.2f}\n' +
               'Train F1-Score: {9:.2f}\n' +
               'Test F1-Score: {10:.2f}\n' +
               'Avg. Epoch Time.: {11:.2f}\n' +
               'Avg. CPU Util.: {12:.2f}\n' +
               'Avg. GPU Util: {13:.2f}')
            .format(
            best_round + 1,
            self.stats['train_loss'][best_round],
            self.stats['test_loss'][best_round],
            self.stats['train_acc'][best_round] * 100,
            self.stats['test_acc'][best_round] * 100,
            self.stats['train_prec'][best_round] * 100,
            self.stats['test_prec'][best_round] * 100,
            self.stats['train_rec'][best_round] * 100,
            self.stats['test_rec'][best_round] * 100,
            self.stats['train_f1'][best_round] * 100,
            self.stats['test_f1'][best_round] * 100,
            sum(self.stats['epoch_time']) / len(self.stats['epoch_time']),
            sum(self.stats['cpu']) / len(self.stats['cpu']),
            sum(self.stats['gpu']) / len(self.stats['gpu'])
        ))

        if show:
            plt.show()

    def start_time(self):
        """
        Starts stopwatch.
        """
        self.time = time()

    def stop_time(self):
        """
        Stops stopwatch and saves time as processing time for an epoch.
        """
        self.epoch_time = time() - self.time
        self.stats['epoch_time'].append(self.epoch_time)

    def add_cpu_usage(self):
        """
        Obtain current CPU utilization
        """
        self.cpu_usage = psutil.cpu_percent()

    def add_gpu_usage(self):
        """
        Obtain current GPU utilization
        """
        self.gpu_usage = nvidia_smi.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu

    def get_performance_metrics(self) -> Tuple:
        """
        Saves intermediate CPU & GPU utilization in summary dict and returns values.
        :return: Tuple with CPU & GPU utilization
        """
        self.stats['cpu'].append(self.cpu_usage)
        self.stats['gpu'].append(self.gpu_usage)
        return self.cpu_usage, self.gpu_usage

    def reset(self):
        """
        Resets intermediate tracking lists and values.
        """
        self.train_losses = []
        self.test_losses = []
        self.train_truth = []
        self.train_pred = []
        self.test_truth = []
        self.test_pred = []

        self.epoch_time = None
        self.cpu_usage = None
        self.gpu_usage = None


class SummaryAgent(object):
    """
    Logs metrics to tensorboard files & makes models persistent.
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
        """
        Saves a PyTorch model to disc.
        :param model: Model to be saved.
        """
        path = os.path.join(self.model_dir, str(self.episode) + '.pt')
        torch.save(model, path)

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
    Intended for use with application context.
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
