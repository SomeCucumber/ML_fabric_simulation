# Math:
import numpy as np

# NN:
import torch
from torch.utils.data import DataLoader, random_split

# Plotting:
from plot_loss import plot_loss

# Files:
import json
import os

# Misc:
import nn as my_models
from loader import Loader
from chunk_gen import ChunkGen
import time

try:
    from iterator_print import percentage_print
    allow_print = True
except ModuleNotFoundError:
    allow_print = False

def main():
    # Device:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: " f"{device}")

    # ===== PARAMETERS =====
    # Model:
    with open("config/model.json") as f:
        model_parameters = json.load(f)

    input_channels = model_parameters["input_channels"]
    output_channels = model_parameters["output_channels"]
    hidden_channels = model_parameters["hidden_channels"]
    dilations = model_parameters["dilations"]
    padding_mode = model_parameters["padding_mode"]
    use_residual_blocks = model_parameters["use_residual_blocks"]
    learning_rate = model_parameters["learning_rate"]
    batch_size = model_parameters["batch_size"]
    max_epochs = model_parameters["max_epochs"]

    # Settings:
    with open("config/settings.json") as f:
        settings = json.load(f)

    model_path = settings["model_path"]
    temp_path = settings["temp_path"]
    # ======================

    model = my_models.CNN(
        input_channels,
        output_channels,
        hidden_channels=hidden_channels,
        use_residual_blocks=use_residual_blocks,
        dilations=dilations,
        padding_mode=padding_mode
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    # Parameter count:
    print("Trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    #  Chunk dataset:
    chunks = ChunkGen()
    print("Dataset chunked...")

    start_time = time.time()
    training_loss_array = np.zeros((chunks.chunk_counter, max_epochs))
    validation_loss_array = np.zeros((chunks.chunk_counter, max_epochs))
    for chunk_nr in range(chunks.chunk_counter):
        # Dataset:
        dataset = Loader(chunk_nr)
        training_set, validation_set = random_split(dataset, [0.8, 0.2])
        train_loader = DataLoader(
            training_set,
            batch_size=batch_size,
            shuffle=True
            )
        validate_loader = DataLoader(
            validation_set,
            batch_size=batch_size,
            shuffle=False
            )

        for epoch in range(max_epochs):
            training_loss_array[chunk_nr, epoch] = my_models.train(model, train_loader, optimizer, criterion, device)
            validation_loss_array[chunk_nr, epoch] = my_models.evaluate(model, validate_loader, criterion, device)

            # Save:
            os.makedirs(model_path, exist_ok=True)
            if epoch % max(1, max_epochs//100) == 0:
                torch.save(model.state_dict(), model_path + f"model_epoch_{epoch}_chunk_{chunk_nr}.pth")
            np.save(model_path + "training_loss.npy", training_loss_array)
            np.save(model_path + "validation_loss.npy", validation_loss_array)

            if allow_print:
                percentage_print((epoch+1)+chunk_nr*max_epochs, (chunks.chunk_counter)*max_epochs, message="Training", start_time=start_time)

        torch.save(model.state_dict(), model_path + f"model_chunk_{chunk_nr}.pth")

    # Save:
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), model_path + "model.pth")

    loss_arrays = [training_loss_array, validation_loss_array]

    plot_loss(loss_arrays, model_path)

    final_training_loss = training_loss_array[-1, -1]
    final_validation_loss = validation_loss_array[-1, -1]
    print(f"Final training loss: {final_training_loss:7.3f}")
    print(f"Final validation loss: {final_validation_loss:5.3f}")

if __name__ == "__main__":
    main()