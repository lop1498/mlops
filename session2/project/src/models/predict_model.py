import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import click
import numpy as np
from baseline import Baseline

@click.command()
@click.argument('path_model', type=click.Path(exists=True))
@click.argument('path_data', type=click.Path())
def main(path_model, path_data):
  model = Baseline(784,10)
  state_dict = torch.load(path_model)
  model.load_state_dict(state_dict) 

  data = torch.tensor(np.load(path_data+'/test_data.npy'), dtype=torch.float)
  data_labels = torch.tensor(np.load(path_data+'/test_labels.npy'))
  dataset = TensorDataset(data, data_labels)
  dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

  images, labels = next(iter(dataloader))
  img = images[0].view(1, 784)
  with torch.no_grad():
      logits = model(img)
  ps = F.softmax(logits, dim=1)
  torch.save(ps, 'probabilities.pt')


if __name__=="__main__":
  main()