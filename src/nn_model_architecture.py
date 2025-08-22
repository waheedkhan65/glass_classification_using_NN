import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
  def __init__(self, in_features=9, h1=64, h2=128, h3=14, out_features=7):
    super().__init__()
    self.fc1 = nn.Linear(in_features, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)
    self.dropout = nn.Dropout(0.3)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)

    return x