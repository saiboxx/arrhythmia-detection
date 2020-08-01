import multiprocessing as mp
import os
from collections import Counter
from typing import Tuple

import numpy as np
import pandas as pd
from pylttb import lttb
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.utils import Timer, load_config, save_pickle


def main():
    """
    Load raw ECG data from disc and transform it to cleansed training data.
    """
    cfg = load_config()

    with Timer('Getting label list'):
        labels, file_list = get_labels(cfg['RAW_DATA_PATH'] + '/Diagnostics.xlsx')

    with Timer('Loading & Downsampling files'):
        ecg_data = get_ecg_data(cfg['RAW_DATA_PATH'] + '/ECGDataDenoised',
                                file_list,
                                cfg['DOWNSAMPLE_THRESHOLD'],
                                cfg['DATA_SLICE'],
                                cfg['NUM_WORKERS'])

    with Timer('Imputing missing values'):
        ecg_data = impute_nans(ecg_data)

    with Timer('Splitting into Train & Test Set'):
        x_train, x_test, y_train, y_test = train_test_split(ecg_data,
                                                            labels,
                                                            test_size=0.2,
                                                            shuffle=True,
                                                            stratify=labels,
                                                            random_state=42)

        print('Final Training set has {} samples'.format(len(x_train)))
        print('Final Test set has {} samples'.format(len(x_test)))
        print('Distribution of labels in Training: {}'.format(Counter(y_train)))
        print('Distribution of labels in Testing: {}'.format(Counter(y_test)))

    with Timer('Normalizing data'):
        x_train, x_test = normalize_data(x_train, x_test)

    with Timer('Saving generated arrays'):
        save_pickle(x_train, cfg['PROCESSED_DATA_DIR'] + '/train_data.pkl')
        save_pickle(y_train, cfg['PROCESSED_DATA_DIR'] + '/train_label.pkl')
        save_pickle(x_test, cfg['PROCESSED_DATA_DIR'] + '/test_data.pkl')
        save_pickle(y_test, cfg['PROCESSED_DATA_DIR'] + '/test_label.pkl')


def get_labels(file_path: str) -> Tuple:
    """
    Loads the list of labels.
    Class labels are mapped to integers. Returns the encoded vector and
    a list of files that contain data matching the encoded versions.
    :param file_path:
    :return: Tuple with encoded target vector and list of files pointing to target.
    """
    df = pd.read_excel(file_path)

    # Indices determined through data exploration
    idx_remove = [64, 308, 1681, 1970, 2436, 2465, 3655, 4588, 5525, 6456, 6656, 7140, 7167, 7344, 8142, 8540,
                  9130, 9386, 9431, 9484, 9486, 9489, 9499, 9747, 9748, 9749, 9750, 9751, 9752, 9753, 9754, 9755,
                  9756, 9757, 9758, 9759, 9760, 9761, 9762, 9763, 9764, 9765, 9766, 9767, 9768, 9769, 9770, 9771,
                  9772, 9773, 9774, 9775, 9776, 9777, 9778, 9779, 9780, 9781, 9782, 9783, 9784, 9785]

    df['FileName'] = df['FileName'].astype('category')
    df = df.sort_values(['FileName'])

    # Same as labels to remove
    drop_label = ['AF', 'AVRT', 'SA', 'SAAWR']
    df = df[~df['Rhythm'].isin(drop_label)]
    df = df.drop(df.index[idx_remove], axis=0)

    file_list = [file + '.csv' for file in df['FileName'].tolist()]
    labels = df['Rhythm'].tolist()

    label_map = []
    for l in labels:
        if l == 'SR':
            lab = 0
        elif l == 'SB':
            lab = 1
        elif l == 'AFIB':
            lab = 2
        else:
            lab = 3
        label_map.append(lab)
    return np.asarray(label_map), file_list


def get_ecg_data(file_dir: str,
                 file_list: list,
                 threshold: int,
                 slice: int,
                 num_worker: int) -> np.ndarray:
    """
    Kicks of a process to load the data files to memory, downsample and concatenate them.
    :param file_dir: Root directory where files are located
    :param file_list: List of files to load
    :param threshold: Target size of timeseries downsampling.
    :param slice: Take first 'slice' values from full timeseries
    :param num_worker: Number of workers to process in parallel
    :return: Datamatrix with shape (num_samples, seq_len, num_features)
    """
    file_list_split = np.array_split(file_list, num_worker)
    pool = mp.Pool(processes=num_worker)
    results = [pool.apply_async(downsample_worker,
                                args=(file_dir,
                                      list(file_list_split[pos]),
                                      threshold,
                                      slice,
                                      pos)) for pos in range(num_worker)]
    output = [p.get() for p in results]
    pool.close()
    pool.join()

    output.sort()
    return np.concatenate([out[1] for out in output])


def downsample_worker(file_dir: str,
                      file_list: list,
                      threshold: int,
                      slice: int,
                      pos: int) -> Tuple[int, np.ndarray]:
    """
    Worker method for loading and downsampling time series files
    :param file_dir: Root directory where files are located
    :param file_list: List of files to load
    :param threshold: Target size of timeseries downsampling.
    :param slice: Take first 'slice' values from full timeseries
    :param pos: index of worker
    :return: Tuple with worker index and datamatrix with shape
    (num_samples, seq_len, num_features)
    """
    ecg_graphs = []
    for file in tqdm(file_list, leave=False):
        ecg = np.genfromtxt(os.path.join(file_dir, file), delimiter=',')
        if len(ecg.shape) != 2:
            print(ecg.shape)
            print(file)
            print('There is a faulty file in your directory! Please remove it.')

        ecg_sampled = downsample(ecg[:slice], threshold)
        ecg_sampled = np.expand_dims(ecg_sampled, axis=0)
        ecg_graphs.append(ecg_sampled)

    return pos, np.concatenate(ecg_graphs)


def downsample(x: np.ndarray, threshold: int) -> np.ndarray:
    """
    Downsamples a given 2D-Array with time series data to a series with the
    length specified in threshold. For downsampling the Largest-Triangle-Three-Buckets
    algorithm is used.
    :param x: 2D Array with time series data
    :param threshold: Target size of time series
    :return: Downsampled time series
    """
    index = np.arange(x.shape[0])
    x_sampled = []
    for i in range(x.shape[1]):
        down_x, down_y = lttb(index, x[:, i], threshold)
        x_sampled.append(down_y)
    return np.vstack(x_sampled).T


def impute_nans(x: np.ndarray) -> np.ndarray:
    """
    Checks if NaNs are contained in the data and if yes imputes them
    with a KNN imputer.
    :param x: Datamatrix
    :return: Datamtrix w/o NaNs
    """
    num_nan = np.count_nonzero(np.isnan(x))
    if num_nan == 0:
        print('No NaNs in the dataset!')
        return x

    for i in range(x.shape[0]):
        x[i] = KNNImputer().fit_transform(x[i])

    assert np.nonzero(~np.isnan(x))
    return x


def normalize_data(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """
    Normalizes the data to have zero mean and unit variance.
    :param train: Training data matrix
    :param test: Test data matrix
    :return: Normalized train and test data
    """
    scalers = {}
    for i in range(train.shape[2]):
        scalers[i] = StandardScaler()
        train[:, :, i] = scalers[i].fit_transform(train[:, :, i])

    for i in range(test.shape[2]):
        test[:, :, i] = scalers[i].transform(test[:, :, i])
    return train, test


if __name__ == '__main__':
    main()
