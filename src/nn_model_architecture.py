import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
  def __init__(self, in_features=9, h1=64, h2=128, h3=64, h4=14,  out_features=7):
    super().__init__()
    self.fc1 = nn.Linear(in_features, h1)
    self.bn1 = nn.BatchNorm1d(h1)

    self.fc2 = nn.Linear(h1, h2)
    self.bn2 = nn.BatchNorm1d(h2)

    self.fc3 = nn.Linear(h2, h3)
    self.bn3 = nn.BatchNorm1d(h3)

    self.fc4 = nn.Linear(h3, h4)
    self.bn4 = nn.BatchNorm1d(h4)

    self.out = nn.Linear(h4, out_features)

    # Dropout layer to prevent overfitting
    self.dropout = nn.Dropout(0.2) 
 

  def forward(self, x):
      x = self.dropout(F.relu(self.bn1(self.fc1(x))))
      x = self.dropout(F.relu(self.bn2(self.fc2(x))))
      x = self.dropout(F.relu(self.bn3(self.fc3(x))))
      x = self.dropout(F.relu(self.bn4(self.fc4(x))))
      x = self.out(x)  

      return x
  


# Alternative model using nn.Sequential for cleaner architecture definition

  class ModelSequential(nn.Module):
    def __init__(self, in_features=9, out_features=7):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(14, out_features)
        )

    def forward(self, x):
        return self.network(x)