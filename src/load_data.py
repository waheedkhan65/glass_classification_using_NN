# Functions for loading & preprocessing data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch
import joblib

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    


def load_data():
    df = pd.read_csv("data\processed\glass_dataset.csv")

    X = df.drop("Type", axis=1).values
    y = df["Type"].values

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # ✅ Save StandardScaler
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Scaler saved to models/scaler.pkl")
    

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long),
    )


