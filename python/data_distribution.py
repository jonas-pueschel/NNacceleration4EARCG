import os
from pathlib import Path

from tqdm import tqdm as tq
from torch.utils.data import DataLoader

from functions import DFTKDataGenerator
import torch
import matplotlib.pyplot as plt
import numpy as np


EPS = np.finfo(np.float32).eps


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


def tv_norm(img):
    """Calculate the total variation of a 2D image.

    Args:
        img: 2D numpy array representing the image.

    Returns:
        tv: float, total variation of the image.
    """
    dx = np.diff(img, axis=1, append=0) ** 2
    dy = np.diff(img, axis=0, append=0) ** 2

    tv = np.sum(np.sqrt(dx + dy))

    return tv


def loss_tv(x, y):
    return tv_norm(x - y)


IMAGE_SIZE = 144
NUM_WORKERS = 6
BATCH_SIZE = 1
INDEX = "dftk_unet"
PREFIX = "partial_"

loss_mse = torch.nn.MSELoss()


current_directory = Path.cwd()
data_path = current_directory / "dftk_data"
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


norm_diff_mse = []
norm_diff_tv = []

train_bar = tq(train_loader)
for x, y in train_bar:
    x_np = x.numpy().reshape([144, 144])
    y_np = y.numpy().reshape([144, 144])

    diff_mse = loss_mse(x, y).item()
    norm_diff_mse.append(diff_mse)

    diff_tv = loss_tv(x_np, y_np)
    norm_diff_tv.append(diff_tv)

plot_distribution(norm_diff_mse, norm_diff_tv)
