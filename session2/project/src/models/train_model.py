import torch
from torch.utils.data import TensorDataset, DataLoader
from baseline import Baseline
import click
import numpy as np


@click.command()
@click.argument('path_data', type=click.Path(exists=True))
def main(path_data):
    # Load the train data and labels
    train_data = torch.tensor(np.load(path_data+'/train_data.npy'), dtype=torch.float)
    train_labels = torch.tensor(np.load(path_data+'/train_labels.npy'))

    # Create a TensorDataset from the train data and labels
    train_dataset = TensorDataset(train_data, train_labels)

    # Create a DataLoader from the train dataset
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Load the test data and labels
    test_data = torch.tensor(np.load(path_data+'/test_data.npy'), dtype=torch.float)
    test_labels = torch.tensor(np.load(path_data+'/test_labels.npy'))

    # Create a TensorDataset from the test data and labels
    # test_dataset = TensorDataset(test_data, test_labels)

    # Create a DataLoader from the test dataset
    # test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    model = Baseline(784, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    epochs = 5
    for epoch in range(epochs):
        loss_epoch = 0
        for images, labels in train_dataloader:
            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
        print(f"Training loss on epoch {epoch}: {loss_epoch/len(train_dataloader)}")

    torch.save(model.state_dict(), 'models/model_epoch{}.pth'.format(epochs))


if __name__ == "__main__":
    # path_data = '/data/processed'
    main()
