import torch
import click

import sys
sys.path.insert(0, "/Users/lop1498/Desktop/MDS/DTU/mlops/mlops/session2/project/src/models")

from baseline import Baseline
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()


@click.command()
@click.argument('path_models', type=click.Path(exists=True))
@click.argument('model_epoch', type=int)
@click.argument('path_data', type=click.Path())
def main(path_models, model_epoch, path_data):
  model = Baseline(784,10)
  state_dict = torch.load('{}/model_epoch{}.pth'.format(path_models, model_epoch))
  model.load_state_dict(state_dict) 

  test_data = torch.tensor(np.load(path_data+'/test_data.npy'), dtype=torch.float)
  test_labels = torch.tensor(np.load(path_data+'/test_labels.npy'))
  test_dataset = TensorDataset(test_data, test_labels)
  test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

  images, labels = next(iter(test_dataloader))
  img = images[0].view(1, 784)
  # Turn off gradients to speed up this part
  with torch.no_grad():
      logits = model(img)

  # Output of the network are log-probabilities, need to take exponential for probabilities
  ps = F.softmax(logits, dim=1)
  view_classify(img.view(1, 28, 28), ps)

if __name__=="__main__":
  main()