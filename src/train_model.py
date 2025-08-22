import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.nn_model_architecture import Model

def train(X_train, y_train, epochs, lr=0.001):
    model = Model()     # Initialize the model
    optimizer = optim.Adam(model.parameters(), lr=lr)     # Optimizer
    loss_fn = nn.CrossEntropyLoss()    # Loss function

    losses = []  # To store losses for plotting

    for epoch in range(epochs): # Loop through epochs
        optimizer.zero_grad()   # Reset gradients
        y_pred = model.forward(X_train)   # Forward pass
        loss = loss_fn(y_pred, y_train)   # Calculate loss
        loss.backward()   # Backward pass
        optimizer.step()    # Update weights
        losses.append(loss.item())  # Store loss

        if (epoch+1) % 10 == 0:      # You can set the value as you prefer
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    plt.plot(range(epochs), losses) # Plotting losses with respect to epochs
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('losses')    


    return model, losses # Return the trained model


def evaluate(model, X_test, y_test):
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        y_pred = model(X_test)
        correct = (y_pred.argmax(1) == y_test).sum().item()
        acc = correct / len(y_test)
        print(f"Accuracy: {acc:.2f}")
