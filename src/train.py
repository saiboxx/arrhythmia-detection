import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.network import TweetDataset, collate_var_sequences, LSTMPredictor, GRUPredictor
from src.utils import load_config, TrackingAgent, SummaryAgent


def main():
    print('Loading config')
    cfg = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Utilizing device {}'.format(device))

    print('Initializing Dataloader & Dataset.')
    training_data = TweetDataset(cfg['PROCESSED_DATA_DIR'])
    data_loader = DataLoader(dataset=training_data,
                             batch_size=cfg['BATCH_SIZE'],
                             shuffle=True,
                             num_workers=4,
                             collate_fn=collate_var_sequences,
                             pin_memory=True)

    print('Initializing Model & Components.')
    model = GRUPredictor(num_classes=training_data.num_classes,
                         hidden_size=cfg['HIDDEN_SIZE'],
                         batch_size=cfg['BATCH_SIZE'],
                         num_layers=cfg['NUM_LAYERS'],
                         drop_out=cfg['DROPOUT']).to(device)
    optimizer = Adam(model.parameters(), lr=cfg['LR'])
    loss = CrossEntropyLoss()

    print('Initializing Helper Agents.')
    tracker = TrackingAgent(cfg['BATCH_SIZE'], len(training_data))
    summary = SummaryAgent(cfg['SUMMARY_PATH'], model.name, cfg)

    print('Start training with {} epochs'.format(cfg['EPOCHS']))
    for e in range(1, cfg['EPOCHS'] + 1):
        for i_batch, sample_batched in enumerate(tqdm(data_loader, leave=False)):
            x_sequence = sample_batched['tweet'].to(device)
            y = sample_batched['label'].to(device)

            y_hat = model(x_sequence)
            batch_loss = loss(y_hat, y)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            tracker.add_loss(batch_loss)
            tracker.add_correct_class(y_hat, y)

        mean_loss = tracker.get_loss()
        accuracy = tracker.get_accuracy()
        tracker.reset()
        summary.add_scalar('Loss', mean_loss)
        summary.add_scalar('Accuracy', accuracy)
        summary.save_model(model)
        summary.adv_episode()
        summary.flush()

        print('Ep. {0}; Mean Epoch Loss {1:.2f}; Train Acc. {2:.2f}'.format(e, mean_loss, accuracy))

    summary.close()


if __name__ == '__main__':
    main()
