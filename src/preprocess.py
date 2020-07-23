import os
from typing import Tuple
import numpy as np
import pandas as pd
from pylttb import lttb
import multiprocessing as mp
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from src.utils import Timer, load_config, save_pickle


def main():
    cfg = load_config()
    col_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    with Timer('Obtaining file list'):
        file_dir = cfg['RAW_DATA_PATH'] + '/ECGDataDenoised'
        file_list = os.listdir(file_dir)
        print('{} files found.'.format(len(file_list)))

    with Timer('Getting label list'):
        labels = get_labels(cfg['RAW_DATA_PATH'] + '/Diagnostics.xlsx', file_list)

    with Timer('Loading & Downsampling files'):
        ecg_data = get_ecg_data(file_dir,
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


def get_labels(file_path: str, file_list: list) -> list:
    df = pd.read_excel(file_path)
    file_list = [file[:-4] for file in file_list]

    df['FileName'] = df['FileName'].astype('category')
    df['FileName'].cat.set_categories(file_list, inplace=True)
    df = df.sort_values(['FileName'])

    labels = df['Rhythm'].tolist()

    label_map = []
    for l in labels:
        if l == 'SR':
            lab = 0
        elif l == 'SB':
            lab = 1
        elif l == 'ST':
            lab = 2
        else:
            lab = 3
        label_map.append(lab)
    return np.asarray(label_map)


def get_ecg_data(file_dir: str,
                 file_list: list,
                 threshold: int,
                 slice: int,
                 num_worker: int) -> np.ndarray:
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


def downsample_worker(file_dir: str, file_list: list, threshold: int, slice: int, pos: int) -> Tuple[int, np.ndarray]:
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
    for i in range(x.shape[0]):
        x[i] = KNNImputer().fit_transform(x[i])

    assert np.nonzero(~np.isnan(x))
    return x


def normalize_data(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    train = scaler.fit_transform(train.reshape(-1, train.shape[-1])).reshape(train.shape)
    test = scaler.transform(test.reshape(-1, test.shape[-1])).reshape(test.shape)
    return train, test


if __name__ == '__main__':
    main()
