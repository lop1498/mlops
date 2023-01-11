from torch import nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, size, classes):
        super().__init__()
        self.size = size
        self.classes = classes
        self.fc1 = nn.Linear(self.size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, self.classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return x
