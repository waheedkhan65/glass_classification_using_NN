from src.train_model import train, evaluate
from src.load_data import load_data, CustomDataset
from src.nn_model_architecture import Model
from torch.utils.data import DataLoader
import torch
import os

if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Create datasets and loaders
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initialize and train
    model = Model()
    trained_model, losses = train(model, train_loader, epochs=100, lr=0.001)

    # Evaluate model
    accuracy = evaluate(trained_model, test_loader)

    # Save the model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/my_nn_model.pt")
    print("Model saved to models/my_nn_model.pt")
