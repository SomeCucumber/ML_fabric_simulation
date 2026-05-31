# Math:
import numpy as np

# Plotting:
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Misc:
import json

def plot_loss(loss_arrays, model_path, single_figure=True, log_scale=True, fontname="Times New Roman") -> None:
    """
    loss_arrays is a list of [training_loss, validation_loss]
    single_figure determines if training and validation loss should be in the same plot
    """
    # Selecting chunks:
    number_of_chunks = np.count_nonzero(loss_arrays[0][:, 0])

    if number_of_chunks > 1:
        selected_chunks = np.linspace(0, number_of_chunks-1, num=5, dtype=int)
    else:
        selected_chunks = None

    # Create figures:
    if single_figure:
        fig, ax = plt.subplots()
    else:
        fig0, ax0 = plt.subplots()
        fig1, ax1 = plt.subplots()

    if single_figure:
        if number_of_chunks > 1:
            for printed_chunk_nr, loss in enumerate(loss_arrays[0][selected_chunks]):
                ax.plot(np.arange(loss.shape[1]), loss, label=f"Training loss, Chunk {selected_chunks[printed_chunk_nr]}")
            for printed_chunk_nr, loss in enumerate(loss_arrays[1][selected_chunks]):
                ax.plot(np.arange(loss.shape[1]), loss, label=f"Validation loss, Chunk {selected_chunks[printed_chunk_nr]}")
        else:
            ax.plot(np.arange(loss_arrays[0].shape[1]), loss_arrays[0][0, :], label="Training loss")
            ax.plot(np.arange(loss_arrays[1].shape[1]), loss_arrays[1][0, :], label="Validation loss")

        finishing(fig, ax, fontname, number_of_chunks, single_figure, log_scale)

    else:
        if number_of_chunks > 1:
            for printed_chunk_nr, loss in enumerate(loss_arrays[0][selected_chunks]):
                ax0.plot(np.arange(loss.shape[1]), loss, label=f"Chunk {selected_chunks[printed_chunk_nr]}")

            for printed_chunk_nr, loss in enumerate(loss_arrays[1][selected_chunks]):
                ax1.plot(np.arange(loss.shape[1]), loss, label=f"Chunk {selected_chunks[printed_chunk_nr]}")

        else:
            ax0.plot(np.arange(loss_arrays[0].shape[1]), loss_arrays[0][0, :])
            ax1.plot(np.arange(loss_arrays[1].shape[1]), loss_arrays[1][0, :])

        finishing(fig0, ax0, fontname, number_of_chunks, single_figure, log_scale, loss_type="Training ")
        finishing(fig1, ax1, fontname, number_of_chunks, single_figure, log_scale, loss_type="Validation ")

def finishing(fig, ax, fontname, number_of_chunks, single_figure, log_scale, loss_type=""):
    ax.set_title(loss_type + "Loss (MSE)", fontname=fontname, fontsize=16)
    ax.set_xlabel("Epoch", fontname=fontname, fontsize=14)
    ax.set_ylabel("Loss", fontname=fontname, fontsize=14)
    if log_scale:
        ax.set_yscale("log")

    if number_of_chunks > 1 or single_figure:
        font_properties = FontProperties(family=fontname, size=11)
        ax.legend(prop=font_properties, loc="upper right")

    fig.tight_layout()
    plt.savefig(model_path + "loss" + ".pdf")

if __name__ == "__main__":
    # ===== PARAMETERS =====
    # Settings:
    with open("config/settings.json") as f:
        settings = json.load(f)

    model_path = settings["model_path"]
    # ======================

    training_loss_array = np.load(model_path + "training_loss.npy")
    validation_loss_array = np.load(model_path + "validation_loss.npy")

    loss_arrays = [training_loss_array, validation_loss_array]

    plot_loss(loss_arrays, model_path)

    final_training_loss = training_loss_array[-1, -1]
    final_validation_loss = validation_loss_array[-1, -1]
    print(f"Final training loss: {final_training_loss:7.3f}")
    print(f"Final validation loss: {final_validation_loss:5.3f}")

    plt.show()