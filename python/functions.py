import re

import matplotlib
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import v2

def plot_losses(training_losses, validation_losses, file_name):
    epochs = range(1, len(training_losses) + 1)

    plt.figure()
    plt.plot(epochs, training_losses, "b", label="Training loss")
    plt.plot(epochs, validation_losses, "r", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(file_name)
    plt.close()


def display_images_with_predictions(data, ground_truth, prediction, file_name):
    num_images = len(data)

    images = [data, prediction, ground_truth]

    # Create the subplots with correct dimensions
    fig, axes = plt.subplots(ncols=num_images, nrows=6, figsize=(num_images, 6))

    for column in range(num_images):
        for row in range(3):
            # Display the first image in the column
            ax = axes[2 * row, column]
            ax.axis("off")
            ax.imshow(images[row][column][0, :, :])

            # Display the second image in the column
            ax = axes[2 * row + 1, column]
            ax.axis("off")
            ax.imshow(images[row][column][1, :, :])

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def display_density_with_predictions(data, ground_truth, prediction, file_name):
    num_images = len(data)

    titles = ["Raw data", "Prediction", "Ground Truth"]
    images = [data, prediction, ground_truth]

    # Number of images per row (assuming num_images is the number of columns for each image)
    num_images = len(images[0])

    # Create the subplots with correct dimensions
    fig, axes = plt.subplots(nrows=3, ncols=num_images, figsize=(num_images, 3))

    # Iterate over the rows and columns to display images and set titles
    for row in range(3):
        for column in range(num_images):
            ax = axes[row, column]
            ax.axis("off")
            density = np.sum(images[row][column] ** 2, axis=0)
            ax.imshow(density)

        # Set the title for the first column in each row
        axes[row, 0].set_title(titles[row], fontsize=12, loc="left")

    # Adjust layout for better spacing
    plt.tight_layout()

    plt.savefig(file_name)
    plt.close()


class DFTKDataGenerator(Dataset):
    def __init__(
        self,
        path: str = None,
        file_names: np.array = None,
        transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])
    ):
        self.base_path = path
        self.file_names = file_names
        self.transforms = transforms

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        # find the corresponding ground_truth file
        pattern = r"partial_(\d+)_(\d+)\.npy"
        match = re.match(pattern, file_name)

        if match:
            file_num = match.group(1)
            percentage = match.group(2)
        else:
            print("Pattern not found in the filename")

        ground_truth_name = f"final_{file_num}.npy"
        grad_name = f"grad_{file_num}_{percentage}.npy"

        gt_path = self.base_path / ground_truth_name
        image_path = self.base_path / file_name
        grad_path = self.base_path / grad_name

        gt_data = torch.from_numpy(np.load(gt_path).astype(np.float32))
        partial = torch.from_numpy(np.load(image_path).astype(np.float32))
        partial_grad = torch.from_numpy(np.load(grad_path).astype(np.float32))

        gt_data = gt_data.permute(2, 0, 1)
        partial = partial.permute(2, 0, 1)
        partial_grad = partial_grad.permute(2, 0, 1)

        all_data = torch.cat((gt_data.unsqueeze(0), partial.unsqueeze(0), partial_grad.unsqueeze(0)), 0)

        all_data = self.transforms(all_data)

        gt_data = all_data[0]
        partial = all_data[1]
        partial_grad = all_data[2]

        combined = np.concatenate((partial, partial_grad), axis=0)

        return combined, gt_data

    def __len__(self):
        return len(self.file_names)
