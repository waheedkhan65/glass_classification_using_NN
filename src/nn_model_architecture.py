import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
  def __init__(self, in_features=9, h1=64, h2=128, h3=14, out_features=7):
    super().__init__()
    self.fc1 = nn.Linear(in_features, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)
    self.dropout = nn.Dropout(0.3)

    # Or we can also do in a easy way like this, but then we have to change the forward method
    '''self.network = nn.sequential(
      nn.Linear(in_features, h1),
      nn.Linear(h1, h2),
      nn.Linear(h2, out_features),
    ) '''

  def forward(self, x):
    # x = self.network(x)    # you will do this if you use nn.sequential otherwise do below
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)

    return x