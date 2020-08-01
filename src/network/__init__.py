from typing import Optional

import torch
from torch import nn, tensor
from torch.utils.data import Dataset

from src.utils import load_pickle


class ECGDataset(Dataset):
    """
    PyTorch Dataset class for ECGs that are preprocessed through "make preprocess".
    """

    def __init__(self, root: str, test: Optional[bool] = False):
        """
        Loads dataset to memory and transforms it to tensor.
        :param root: Directory where data files are located
        """
        self.root = root

        if test:
            self.data = tensor(load_pickle(root + '/test_data.pkl'), dtype=torch.float)
            self.label = tensor(load_pickle(root + '/test_label.pkl'), dtype=torch.long)
        else:
            self.data = tensor(load_pickle(root + '/train_data.pkl'), dtype=torch.float)
            self.label = tensor(load_pickle(root + '/train_label.pkl'), dtype=torch.long)

        self.num_classes = int(max(self.label) + 1)

    def __len__(self) -> int:
        """
        Returns number of samples in dataset.
        :return: number of samples in dataset
        """
        return len(self.label)

    def __getitem__(self, idx: int) -> dict:
        """
        Given an index, return the corresponding data pair.
        :param idx: Index of entry in dataset
        :return: Dict w/ sequence and label.
        """
        return {'data': self.data[idx], 'label': self.label[idx]}


class ModelFactory(object):
    """
    Factory that returns the model described in the configuration.
    """

    def __init__(self, config: dict, num_classes: int, input_size: int, input_length: int):
        """
        Initialize a model factory.
        :param config: Config file as dictionary
        :param num_classes: Number of classes in the dataset
        :param input_size: Number of input features
        :param input_length: Length of input sequence
        """
        self.config = config
        self.num_classes = num_classes
        self.input_size = input_size
        self.input_length = input_length

    def get(self) -> nn.Module:
        """
        Get a ready-to-train PyTorch model.
        Depending on the choice in the config file a LSTM, GRU or CNN is returned.
        :return: A PyTorch module
        """
        model_name = self.config['MODEL']

        if model_name == 'LSTM':
            model = LSTMPredictor(num_classes=self.num_classes,
                                  input_size=self.input_size,
                                  hidden_size=self.config['HIDDEN_SIZE'],
                                  batch_size=self.config['BATCH_SIZE'],
                                  num_layers=self.config['NUM_LAYERS'],
                                  drop_out=self.config['DROPOUT'])
        elif model_name == 'GRU':
            model = GRUPredictor(num_classes=self.num_classes,
                                 input_size=self.input_size,
                                 hidden_size=self.config['HIDDEN_SIZE'],
                                 batch_size=self.config['BATCH_SIZE'],
                                 num_layers=self.config['NUM_LAYERS'],
                                 drop_out=self.config['DROPOUT'])

        elif model_name == 'CNN':
            model = CNNPredictor(num_classes=self.num_classes,
                                 input_size=self.input_size,
                                 input_length=self.input_length,
                                 batch_size=self.config['BATCH_SIZE'])

        else:
            raise ValueError('Supplied model is no valid option!')

        num_params = sum(p.numel() for p in model.parameters())
        print('{0} model has {1} parameters.'.format(model_name, num_params))

        return model


class LSTMPredictor(nn.Module):
    """
    LSTM model for classifying sequences.
    """

    def __init__(self,
                 num_classes: int,
                 input_size: int,
                 hidden_size: int,
                 batch_size: int,
                 num_layers: int,
                 drop_out: int):
        """
        Initialize a LSTM model.
        :param num_classes: Number of target classes
        :param input_size: Number of input features
        :param hidden_size: Size of hidden dimensions
        :param batch_size: Batch size used for training
        :param num_layers: Number of LSTM layers.
        :param drop_out: Dropout probability between LSTM layers.
        """
        super(LSTMPredictor, self).__init__()
        self.name = "LSTM"
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.num_layers = num_layers
        self.drop_out = drop_out
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=self.drop_out,
                            bidirectional=False)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: tensor) -> tensor:
        """
        Propagate a tensor through the model
        :param x: Training data with shape (batch_size, seq_len, num_features)
        :return: Class scores with shape (batch_size, num_classes)
        """
        output, (hidden, cell) = self.lstm(x)
        return self.fc(hidden[-1])


class CNNPredictor(nn.Module):
    """
    CNN model for classifying sequences.
    """

    def __init__(self,
                 num_classes: int,
                 input_size: int,
                 batch_size: int,
                 input_length: int):
        """
        Initialize a CNN model.
        :param num_classes: Number of target classes
        :param input_size: Number of input features
        :param batch_size: Batch size used for training
        :param input_length: Length of input sequence
        """
        super(CNNPredictor, self).__init__()
        self.name = "CNN"
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.input_size = input_size
        self.input_length = input_length

        self.filters = 32
        self.num_layers = 5

        self.conv_layers = [
            nn.Sequential(
                nn.Conv1d(in_channels=self.input_size,
                          out_channels=self.filters,
                          kernel_size=5,
                          stride=2),
                nn.LeakyReLU(),
                nn.MaxPool1d(2)
            )]

        for _ in range(self.num_layers - 2):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=self.filters,
                              out_channels=self.filters,
                              kernel_size=3,
                              stride=1),
                    nn.LeakyReLU(),
                    nn.MaxPool1d(2)
                )
            )

        self.conv_layers.append(
            nn.Conv1d(in_channels=self.filters,
                      out_channels=self.num_classes,
                      kernel_size=3)
        )

        self.conv_layers = nn.Sequential(*self.conv_layers)

    def forward(self, x: tensor) -> tensor:
        """
        Propagate a tensor through the model
        :param x: Training data with shape (batch_size, seq_len, num_features)
        :return: Class scores with shape (batch_size, num_classes)
        """
        x = x.permute(0, 2, 1)
        for layer in self.conv_layers:
            x = layer(x)
        return torch.mean(x, dim=2)


class GRUPredictor(nn.Module):
    """
    GRU model for classifying sequences.
    """

    def __init__(self,
                 num_classes: int,
                 input_size: int,
                 hidden_size: int,
                 batch_size: int,
                 num_layers: int,
                 drop_out: int):
        """
        Initialize a GRU model.
        :param num_classes: Number of target classes
        :param input_size: Number of input features
        :param hidden_size: Size of hidden dimensions
        :param batch_size: Batch size used for training
        :param num_layers: Number of GRU layers.
        :param drop_out: Dropout probability between GRU layers.
        """
        super(GRUPredictor, self).__init__()
        self.name = "GRU"
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.num_layers = num_layers
        self.drop_out = drop_out
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True,
                          dropout=self.drop_out,
                          bidirectional=False)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: tensor) -> tensor:
        """
        Propagate a tensor through the model
        :param x: Training data with shape (batch_size, seq_len, num_features)
        :return: Class scores with shape (batch_size, num_classes)
        """
        output, hidden = self.gru(x)
        return self.fc(hidden[-1])
