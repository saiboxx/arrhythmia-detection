# ECG Arrhythmia Detection and Classification

This project aims to examine recurrent and convolutional approaches for identifying arrhythmias in ECGs.
An extended project report is available [here](report/IAAML_arrythmia_detection.pdf). The data can be downloaded [here](https://figshare.com/collections/ChapmanECG/4560497/2).
Relevant source code is in 'src'.

## How to use

1. Clone the project
    ```
    git clone https://github.com/saiboxx/arrhythmia-detection.git
    ```
2. I recommend to create an own python virtual environment and activate it:
    ```
    cd offline-reinforcement-learning
    python -m venv .venv
    source .venv/bin/activate
    ```
3. Install necessary packages
    ```
    make requirements
    ```
4. [Download](https://figshare.com/collections/ChapmanECG/4560497/2) the Denoised data and the Diagnostics and place it in 'data/raw'.
3. Preprocess data
    ```
    make preprocess
    ```
3. Train a model
    ```
    make train
    ```
   
4. Track training via tensorboard
    ```
    tensorboard --logdir=logs
    ```
    
5. Enjoy your model. After training your model will be in '/models'. Plots are in '/plots'.
6. Make some predictions. If you have some data, where you want to apply your model, enter the relevant params in the config and start inference.
    ```
    make inference
    ```
   
 ## Parameters
 
 This projects comes with a global config file, which enables convenient changing of parameters. In most cases it is fine to leave the defaults.
 The 'config.yml' contains following adjustable parameters:
 
**Preprocessing:**
 - RAW_DATA_PATH: Path pointing to raw data
 - PROCESSED_DATA_DIR: Path to directory, which should contain the processed data.
 - DATA_SLICE: Takes the first _n_ elements from the whole time series
 - NUM_WORKERS: Number of workers for preprocessing.
 - DOWNSAMPLE_THRESHOLD: Target size for downsampling

**Training:**

 - SUMMARY_PATH: Root directory for Tensorboard files
 - MODEL: Model to train. Choose between LSTM, GRU or CNN.
 - EPOCHS: Epochs to train
 - BATCH_SIZE: Batch size
 - HIDDEN_SIZE: Hidden size for LSTM or GRU
 - NUM_LAYERS: Stacked LSTM or GRU layers.
 - DROPOUT: Probability for values to be zeroed between LSTM or GRU units.
 - LR: Learning Rate

**Inference:**
 - INFERENCE_MODEL: Path to model
 - INFERENCE_DATA: Path to inference data
 - INFERENCE_LABEL: Path to inference label (if given). Can be left blank.
