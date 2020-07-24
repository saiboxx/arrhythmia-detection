import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.network import ECGDataset, ModelFactory
from src.utils import load_config, TrackingAgent, SummaryAgent


def main():
    print('Loading config')
    cfg = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Utilizing device {}'.format(device))

    print('Initializing Dataloader & Dataset.')
    train_data = ECGDataset(cfg['PROCESSED_DATA_DIR'])
    test_data = ECGDataset(cfg['PROCESSED_DATA_DIR'], test=True)
    train_loader = DataLoader(dataset=train_data,
                              batch_size=cfg['BATCH_SIZE'],
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(dataset=test_data,
                             batch_size=cfg['BATCH_SIZE'],
                             num_workers=4,
                             pin_memory=True)

    print('Initializing Model & Components.')
    model_factory = ModelFactory(cfg,
                                 num_classes=train_data.num_classes,
                                 input_size=train_data.data.shape[2],
                                 input_length=train_data.data.shape[1])

    model = model_factory.get().to(device)
    optimizer = Adam(model.parameters(), lr=cfg['LR'])
    loss = CrossEntropyLoss()

    print('Initializing Helper Agents.')
    tracker = TrackingAgent(cfg['BATCH_SIZE'], len(train_data))
    summary = SummaryAgent(cfg['SUMMARY_PATH'], model.name, cfg)

    print('Start training with {} epochs'.format(cfg['EPOCHS']))
    for e in range(1, cfg['EPOCHS'] + 1):

        tracker.start_time()
        model.train()
        for i_batch, sample_batched in enumerate(tqdm(train_loader, leave=False)):
            x_sequence = sample_batched['data'].to(device)
            y = sample_batched['label'].to(device)

            y_hat = model(x_sequence)
            batch_loss = loss(y_hat, y)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            tracker.add_train_loss(batch_loss)
            tracker.add_train_prediction(y_hat, y)

        tracker.stop_time()
        tracker.add_cpu_usage()
        tracker.add_gpu_usage()

        model.eval()
        for i_batch, sample_batched in enumerate(test_loader):
            with torch.no_grad():
                x_test = sample_batched['data'].to(device)
                y_test = sample_batched['label'].to(device)
                y_hat = model(x_test)
                test_loss = loss(y_hat, y_test)

            tracker.add_test_loss(test_loss)
            tracker.add_test_prediction(y_hat, y_test)

        train_metrics = tracker.get_train_metrics()
        train_loss = tracker.get_train_loss()
        summary.add_scalar('Train Loss', train_loss)
        summary.add_scalar('Train Accuracy', train_metrics[0])
        summary.add_scalar('Train Precision', train_metrics[1])
        summary.add_scalar('Train Recall', train_metrics[2])
        summary.add_scalar('Train F1-Score', train_metrics[3])

        test_metrics = tracker.get_test_metrics()
        test_loss = tracker.get_test_loss()
        summary.add_scalar('Test Loss', test_loss)
        summary.add_scalar('Test Accuracy', test_metrics[0])
        summary.add_scalar('Test Precision', test_metrics[1])
        summary.add_scalar('Test Recall', test_metrics[2])
        summary.add_scalar('Test F1-Score', test_metrics[3])

        cpu, gpu = tracker.get_performance_metrics()
        summary.add_scalar('CPU Utilization', cpu)
        summary.add_scalar('GPU Utilization', gpu)
        summary.add_scalar('Epoch Time', tracker.epoch_time)


        tracker.reset()
        summary.save_model(model)
        summary.adv_episode()
        summary.flush()

        print('Ep. {0}; Epoch Loss {1:.2f}; Train Acc. {2:.2f}; Val. Loss {3:.2f}; Val. Acc. {4:.2f}'
              .format(e, train_loss, train_metrics[0], test_loss, test_metrics[0]))

    summary.close()
    tracker.get_plots(show=True)


if __name__ == '__main__':
    main()
