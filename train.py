# Math:
import numpy as np

# NN:
import torch
from torch.utils.data import DataLoader, random_split

# Plotting:
import matplotlib.pyplot as plt

# Misc:
import nn as my_models
from loader import Loader
import json

# Device:
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: " f"{device}")

# ===== PARAMETERS =====
# Model:
with open("config/model.json") as f:
    model_parameters = json.load(f)

network = model_parameters["network"]
network_args = model_parameters["network_args"]
learning_rate = model_parameters["learning_rate"]
batch_size = model_parameters["batch_size"]
max_epochs = model_parameters["max_epochs"]

# Simulation:
with open("config/simulation.json") as f:
    simulation_parameters = json.load(f)

N_x = simulation_parameters["N_x"]
N_y = simulation_parameters["N_y"]
dt = simulation_parameters["dt"]

# Settings:
with open("config/settings.json") as f:
    settings = json.load(f)

model_path = settings["model_path"]
# ======================

# Plotting:
def plot_loss(loss_list, font="Times New Roman") -> None:
    fig, ax = plt.subplots()
    ax.plot(range(len(loss_list)), loss_list)
    ax.set_title("Validation loss (MSE)", fontname=font, fontsize=16)
    ax.set_xlabel("Epoch", fontname=font, fontsize=14)
    ax.set_ylabel("Loss", fontname=font, fontsize=14)
    fig.tight_layout()
    plt.show()

# Dataset:
dataset = Loader(N_x, N_y, dt, device)
training_set, validation_set = random_split(dataset, [0.8, 0.2])
train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
validate_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

# Architectures:
architecture = {
    "CNN": my_models.CNN
    }

model = architecture[network](*network_args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

validation_loss_list = []
for epoch in range(max_epochs):
    _ = my_models.train(model, train_loader, optimizer, criterion, device)
    validation_loss_list.append(my_models.evaluate(model, validate_loader, criterion, device))

    print(f"Epoch {epoch+1}/{max_epochs}")

# Save model:
torch.save(model.state_dict(), model_path)

plot_loss(validation_loss_list)