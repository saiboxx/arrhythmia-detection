from typing import Optional

import torch
from torch import nn, tensor
from torch.utils.data import Dataset

from src.utils import load_pickle


class ECGDataset(Dataset):
    """
    PyTorch Dataset class for tweets that are preprocessed through "make preprocess".
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
    def __init__(self, config: dict, num_classes: int, input_size: int, input_length: int):
        self.config = config
        self.num_classes = num_classes
        self.input_size = input_size
        self.input_length = input_length

    def get(self):
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
    def __init__(self,
                 num_classes: int,
                 input_size: int,
                 hidden_size: int,
                 batch_size: int,
                 num_layers: int,
                 drop_out: int):
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
        output, (hidden, cell) = self.lstm(x)
        x = self.fc(hidden[-1])
        return x


class CNNPredictor(nn.Module):
    def __init__(self,
                 num_classes: int,
                 input_size: int,
                 batch_size: int,
                 input_length: int):
        super(CNNPredictor, self).__init__()
        self.name = "CNN"
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.input_size = input_size
        self.input_length = input_length

        self.pool = nn.MaxPool1d(2)
        self.elu = nn.ELU()

        self.conv1 = nn.Conv1d(in_channels=self.input_size,
                               out_channels=128,
                               kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=128,
                               out_channels=64,
                               kernel_size=5)
        self.conv3 = nn.Conv1d(in_channels=64,
                               out_channels=32,
                               kernel_size=3)

        self.fc_size = self.get_fc_size()

        self.fc1 = nn.Linear(in_features=self.fc_size, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=self.num_classes)

    def get_fc_size(self):
        with torch.no_grad():
            x = torch.ones((1, self.input_size, self.input_length))
            x = self.conv1(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.pool(x)
            x = self.conv3(x)
            x = self.pool(x)
            return len(x.flatten())

    def forward(self, x: tensor) -> tensor:
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.elu(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 22)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        return x


class GRUPredictor(nn.Module):
    def __init__(self,
                 num_classes: int,
                 input_size: int,
                 hidden_size: int,
                 batch_size: int,
                 num_layers: int,
                 drop_out: int):
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
        output, hidden = self.gru(x)
        x = self.fc(hidden[-1])
        return x
