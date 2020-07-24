import torch
from torch import tensor
from sklearn.metrics import classification_report, cohen_kappa_score
from src.utils import load_config, load_pickle
from src.network import LSTMPredictor, GRUPredictor, CNNPredictor


def main():
    cfg = load_config()

    print('Loading Model & Data.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(cfg['INFERENCE_MODEL']).to(device)
    model.eval()
    data = tensor(load_pickle(cfg['INFERENCE_DATA'])).float().to(device)
    label = None
    if len(cfg['INFERENCE_DATA']) > 0:
        label = load_pickle(cfg['INFERENCE_LABEL'])

    print('Starting Inference.')
    with torch.no_grad():
        y_hat = model(data)
        y_hat = torch.argmax(y_hat, dim=1).tolist()

    if label is not None:
        print('Generating Report')
        print(classification_report(label, y_hat, target_names=['SR', 'SB', 'ST', 'Other']))
        print('Kappe Score: {:.2f}'.format(cohen_kappa_score(label, y_hat)))


if __name__ == '__main__':
    main()
