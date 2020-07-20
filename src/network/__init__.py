from typing import Tuple

import torch
from torch import nn, tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence

from src.utils import load_pickle


class TweetDataset(Dataset):
    """
    PyTorch Dataset class for tweets that are preprocessed through "make preprocess".
    """

    def __init__(self, root: str):
        """
        Loads dataset to memory and transforms it to tensor.
        :param root: Directory where data files are located
        """
        self.root = root

        self.train = load_pickle(root + '/train_medium.pkl')
        self.label = torch.tensor(load_pickle(root + '/label_medium.pkl'))

        self.num_classes = max(max(self.train)) + 1
        self.ohe_mapping = torch.eye(self.num_classes)

    def __len__(self) -> int:
        """
        Returns number of samples in dataset.
        :return: number of samples in dataset
        """
        return len(self.label)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Given an index, return the corresponding one-hot encoded data pair.
        :param idx: Index of entry in dataset
        :return: Tuple w/ sequence as one hot encoding and label.
        """
        tweet = self.ohe_mapping[self.train[idx]]
        return tweet, self.label[idx]


def collate_var_sequences(batch: list) -> dict:
    """
    Custom collate function to handle sequences of varying length.
    :param batch: List of samples obtained from Dataset
    :return: Collated dict of samples.
    """
    tweets = pack_sequence(sequences=[b[0] for b in batch], enforce_sorted=False)
    label = torch.stack([b[1] for b in batch])
    return {'tweet': tweets, 'label': label}


class LSTMPredictor(nn.Module):
    def __init__(self,
                 num_classes: int,
                 hidden_size: int,
                 batch_size: int,
                 num_layers: int,
                 drop_out: int):
        super(LSTMPredictor, self).__init__()
        self.name = "LSTM"
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.num_layers = num_layers
        self.drop_out = drop_out
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lstm = nn.LSTM(input_size=self.num_classes,
                            hidden_size=hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=self.drop_out,
                            bidirectional=False)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: tensor) -> tensor:
        output, (hidden, cell) = self.lstm(x)
        x = self.fc(hidden[-1])
        return x


class GRUPredictor(nn.Module):
    def __init__(self,
                 num_classes: int,
                 hidden_size: int,
                 batch_size: int,
                 num_layers: int,
                 drop_out: int):
        super(GRUPredictor, self).__init__()
        self.name = "GRU"
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.num_layers = num_layers
        self.drop_out = drop_out
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gru = nn.GRU(input_size=self.num_classes,
                          hidden_size=hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True,
                          dropout=self.drop_out,
                          bidirectional=False)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: tensor) -> tensor:
        output, hidden = self.gru(x)
        x = self.fc(hidden[-1])
        return x
