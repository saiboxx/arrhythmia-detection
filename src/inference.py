import torch
import numpy as np
from src.network import LSTMPredictor
from src.utils import load_config, load_pickle


def main():
    cfg = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Utilizing device {}'.format(device))

    map_char_to_int = load_pickle(cfg['PROCESSED_DATA_DIR'] + '/map_char_to_int.pkl')
    map_int_to_char = load_pickle(cfg['PROCESSED_DATA_DIR'] + '/map_int_to_char.pkl')
    num_chars = len(map_int_to_char)
    ohe_mapping = torch.eye(num_chars).to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Utilizing device {}'.format(device))

    print('Loading Model')
    model = LSTMPredictor(num_chars, cfg['HIDDEN_SIZE'], cfg['BATCH_SIZE']).to(device)
    model.load_state_dict(torch.load(cfg['MODEL_PATH']))
    model.eval()

    while True:
        input_str = input('Tweet: ')
        if input_str == "exit":
            break

        input_str = input_str.lower()
        while True:
            input_enc = [map_char_to_int[char] for char in list(input_str)]
            input_ohe = ohe_mapping[input_enc].unsqueeze(dim=0)
            with torch.no_grad():
                output = model(input_ohe)
                output_prob = torch.softmax(output, dim=1).squeeze().cpu().numpy()

            result = np.random.choice(num_chars, p=output_prob)
            input_str += map_int_to_char[result]

            if len(input_str) >= 140 or (len(input_str) >= 100 and input_str[-1] == "."):
                print(input_str)
                break


if __name__ == '__main__':
    main()
