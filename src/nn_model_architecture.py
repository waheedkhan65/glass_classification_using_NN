import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
  def __init__(self, in_features=9, h1=64, h2=128, h3=128, h4=64, h5=14,  out_features=7):
    super().__init__()
    self.fc1 = nn.Linear(in_features, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.fc3 = nn.Linear(h2, h3)
    self.fc4 = nn.Linear(h3, h4)
    self.fc5 = nn.Linear(h4, h5)
    self.out = nn.Linear(h5, out_features)
    self.dropout = nn.Dropout(0.2) # Dropout layer to prevent overfitting

    # You can also use nn.Sequential to define the network in a more compact way
    # but then you will have to change the forward method accordingly
    '''self.network = nn.sequential(
      nn.Linear(in_features, h1),
      nn.Linear(h1, h2),
      nn.Linear(h2, out_features),
    ) '''

  def forward(self, x):
    # x = self.network(x)    # you will do this if you use nn.sequential otherwise do below
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    x = F.relu(self.fc5(x))
    x = self.out(x)

    return x