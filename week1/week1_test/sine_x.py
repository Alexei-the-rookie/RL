import torch
import numpy as np
import torch.nn as nn
import matplotlib as plt
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class SineNet(nn.Module):
    def __init__(self, n_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, x):
        return self.net(x)
    
model=SineNet()

learning_rate = 1e-3
batch_size = 64
epochs = 90
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

x = torch.linspace(-2*np.pi, 2*np.pi, 1000).unsqueeze(1)
y = torch.sin(x)
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
loss_function = nn.MSELoss()

def training_loop(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    model.train()
    epoch_loss = 0  # Accumulate loss for the entire epoch
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_function(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()  # Accumulate batch loss
        current = (batch + 1) * len(X)  # Correctly calculate the number of processed samples
        if batch % 3 == 0:  # Print every 10 batches
            avg_loss = epoch_loss / (batch + 1)  # Calculate average loss up to the current batch
            print(f"avg_loss: {avg_loss:>7f}  [{current:>5d}/{size:>5d}]")

x_test = torch.linspace(-3 * np.pi, 3 * np.pi, 1000).unsqueeze(1)
y_test = torch.sin(x_test)

def test_loop(dataloader, model, loss_function):
    model.eval()
    with torch.no_grad():
        y_pred_test = model(x_test)
    test_loss = loss_function(y_pred_test, y_test)
    print(f"Test Error: {test_loss:>8f}")
    return y_pred_test

test_losses = []  # List to store test losses

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    training_loop(dataloader, model, loss_function, optimizer)
    y_pred_test = test_loop(dataloader, model, loss_function)
    test_loss = loss_function(y_pred_test, y_test).item()
    test_losses.append(test_loss)  # Record test loss

# Plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Subplot 1: Fitted curve
ax1.plot(x_test.numpy(), y_test.numpy(), label='True Sine Function', color='blue')
ax1.plot(x_test.numpy(), y_pred_test.numpy(), label='Fitted Curve', color='red', linestyle='--')
ax1.set_title('Sine Function Fitting')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()
ax1.grid()

# Subplot 2: Test loss curve
ax2.plot(range(1, epochs + 1), test_losses, label='Test Loss', color='green')
ax2.set_title('Test Loss Over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.show()

print("Done!")