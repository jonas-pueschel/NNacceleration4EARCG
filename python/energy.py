import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from functions import DFTKDataGenerator


def plot_distribution(data1, data2):
    bins = 10

    # Create a figure with two subplots (2 rows, 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Plot the first histogram on the first subplot
    ax1.hist(data1, bins=bins, alpha=0.7, color="blue")
    ax1.set_title("Histogram of mse")
    ax1.set_ylabel("Frequency")

    # Plot the second histogram on the second subplot
    ax2.hist(data2, bins=bins, alpha=0.7, color="red")
    ax2.set_title("Histogram of tv")
    ax2.set_ylabel("Frequency")

    # Adjust layout for better spacing
    plt.tight_layout()

    # Display the plot
    plt.show()


IMAGE_SIZE = 144
NUM_WORKERS = 6
BATCH_SIZE = 1
INDEX = "dftk_unet"
PREFIX = "partial_"

loss_mse = torch.nn.MSELoss()


current_directory = Path.cwd().parent
data_path = current_directory / "data"
training_data_path = data_path / "training"

# find all files that start with "partial_" in the directory that contains the training data
all_training_files = os.listdir(training_data_path)
training_files = [file for file in all_training_files if file.startswith(PREFIX)]


train_dataset = DFTKDataGenerator(path=training_data_path, file_names=training_files)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


def norm(array):
    squared_sum = np.sum(array**2, axis=-1)
    total_sum = np.sum(squared_sum)
    return total_sum


norm_diff_mse = []
norm_diff_tv = []

for x, y in train_loader:
    x_np = x.numpy().reshape([144, 144, 2])
    y_np = y.numpy().reshape([144, 144, 2])

    norm_1 = norm(x_np)
    norm_2 = norm(y_np)
    print(norm_1)
    print(norm_2)
