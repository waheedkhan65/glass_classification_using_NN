import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def train(model, train_loader, epochs, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_feature, batch_label in train_loader:
            optimizer.zero_grad()  # Clear gradients
            y_pred = model(batch_feature)  # Forward pass
            loss = loss_fn(y_pred, batch_label)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            epoch_loss += loss.item()  # Accumulate loss

        losses.append(epoch_loss / len(train_loader))   # Average loss for the epoch

        if (epoch + 1) % 10 == 0:  # Print every 10 epochs, you can also change it 
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Plot training loss
    plt.plot(range(epochs), losses)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    return model, losses

# Evaluate function
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            y_pred = model(batch_features)
            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)

    acc = correct / total
    print(f"Accuracy: {acc:.2f}")
    return acc
