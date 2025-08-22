from src.train_model import train, evaluate
from src.load_data import load_data
import torch
import os

if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Train model
    model, losses = train(X_train, y_train, epochs=100, lr=0.001)

    # Evaluate model
    evaluate(model, X_test, y_test)

    #Save the model
    os.makedirs("models", exist_ok=True)       # Create the models folder if it doesn't exist
    torch.save(model.state_dict(),"models/my_nn_model.pt" )
    print("Model saved to models/my_nn_model.pt")
