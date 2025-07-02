import math
import os
import json
from pathlib import Path

import torch
from functions import DFTKDataGenerator, display_density_with_predictions
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from unet.model import UNet
import datetime


def save_model_checkpoint(model, checkpoint_name):
    torch.save(model.state_dict(), os.path.join(working_directory, checkpoint_name))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Load model from saved checkpoint
def load_model_from_checkpoint(model, ckp_path):
    return model.load_state_dict(
        torch.load(
            ckp_path,
            map_location=get_device(),
        )
    )


# Send the Tensor or Model (input argument x) to the right device
# for this notebook. i.e. if GPU is enabled, then send to GPU/CUDA
# otherwise send to CPU.
def to_device(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x.cpu()


def get_model_parameters(m):
    total_params = sum(param.numel() for param in m.parameters())
    return total_params


def print_model_parameters(m):
    num_model_parameters = get_model_parameters(m)
    print(f"The Model has {num_model_parameters/1e6:.2f}M parameters")


def close_figures():
    while len(plt.get_fignums()) > 0:
        plt.close()


# Train the model for a single epoch
def train_model(model, loader, optimizer, loss_function):
    training_losses = []

    for batch_idx, (inputs, targets) in enumerate(loader, 0):
        optimizer.zero_grad()
        inputs = to_device(inputs)
        targets = to_device(targets)
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        training_losses.append(loss.item())

    losses_tensor = torch.FloatTensor(training_losses)
    return losses_tensor.mean()


def test_dataset_accuracy(model, loader, criterion):
    losses = []
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = to_device(inputs)
        targets = to_device(targets)
        predictions = model(inputs)

        loss = criterion(predictions, targets)
        losses.append(loss.item())

        del inputs
        del targets
        del predictions

    losses_tensor = torch.FloatTensor(losses)
    return losses_tensor.mean()


def sample_images(model, inputs, targets, image_path):
    inputs = to_device(inputs)
    predictions = model(inputs)

    inputs = inputs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    outputs = predictions.detach().cpu().numpy()

    display_density_with_predictions(inputs, targets, outputs, image_path)


def plot_and_save_losses(train_losses, val_losses, save_path):
    """
    Plots training and validation losses and saves the plot to the specified path.

    Parameters:
    - train_losses: List of training losses.
    - val_losses: List of validation losses.
    - save_path: Path where the plot image will be saved.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Losses")
    plt.savefig(save_path)
    plt.close()


def save_loss_vector(train_losses, val_losses, file):
    losses = {
        "val": val_losses,
        "train": train_losses,
    }
    with open(file, "w") as f:
        json.dump(losses, f)


def train_loop(
    model, loader, test_data, epochs, optimizer, loss_function, scheduler, save_path
):
    best_accuarcy = math.inf
    test_inputs, test_targets = test_data
    training_losses = []
    validation_losses = []
    epoch_i, epoch_j = epochs
    to_device(model)

    for epoch in range(epoch_i, epoch_j):
        print(f"Epoch: {epoch:02d}, Learning Rate: {optimizer.param_groups[0]['lr']}")
        model.train()

        train_model(model, loader, optimizer, loss_function)

        model.eval()
        with torch.inference_mode():
            training_loss = test_dataset_accuracy(model, loader_training, loss_function)
            validation_loss = test_dataset_accuracy(
                model, loader_validation, loss_function
            )
            training_losses.append(float(training_loss))
            validation_losses.append(float(validation_loss))

            if validation_loss < best_accuarcy:
                best_accuarcy = validation_loss
                print(f"New lowest loss in epoch {epoch}; Saving model")
                torch.save(model.state_dict(), save_path / "dftk_model.pth")

            train_path = save_path / f"epoch_{epoch}_training.png"
            test_path = save_path / f"epoch_{epoch}_validation.png"
            sample_images(model, train_sample_inputs, train_sample_targets, train_path)
            sample_images(model, test_sample_inputs, test_sample_targets, test_path)

        if scheduler is not None:
            scheduler.step()

    return training_losses, validation_losses


if __name__ == "__main__":
    current_time = datetime.datetime.now()

    IMAGE_SIZE = 192 #144 for a = 15, 192 for a = 20 TODO: read this from data
    NUM_WORKERS = 6
    BATCH_SIZE = 20
    BATCH_SIZE_EVAL = 8
    #name of the directory of the unet
    INDEX = f"unet_{current_time:%Y-%m-%d_%H-%M-%S}"
    LR = 0.0001
    EPOCHS = 100
    PREFIX = "partial_"

    working_directory = Path.cwd()
    # Validation: Check if CUDA is available
    print(f"CUDA: {torch.cuda.is_available()}")

    data_path = working_directory / "data"
    training_data_path = data_path / "training"
    validation_data_path = data_path / "evaluation"

    # find all files that start with "partial_" in the directory that contains the training data
    all_training_files = os.listdir(training_data_path)
    training_files = [file for file in all_training_files if file.startswith(PREFIX)]

    all_evaluation_files = os.listdir(validation_data_path)
    evaluation_files = [
        file for file in all_evaluation_files if file.startswith(PREFIX)
    ]

    train_dataset = DFTKDataGenerator(
        path=training_data_path, file_names=training_files
    )
    valid_dataset = DFTKDataGenerator(
        path=validation_data_path,
        file_names=evaluation_files,
    )

    loader_training = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    loader_validation = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    (train_sample_inputs, train_sample_targets) = next(iter(loader_training))
    (test_sample_inputs, test_sample_targets) = next(iter(loader_validation))

    model = UNet(out_channels=2, in_channels=4)
    to_device(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.7)

    save_path = working_directory / "results" / INDEX
    save_path.mkdir(parents=True, exist_ok=True)

    loss = torch.nn.MSELoss()

    trn_loss, val_loss = train_loop(
        model,
        loader_training,
        (test_sample_inputs, test_sample_targets),
        (0, EPOCHS),
        optimizer,
        loss,
        scheduler,
        save_path,
    )
    plot_and_save_losses(trn_loss, val_loss, save_path / "losses.pdf")
    save_loss_vector(trn_loss, val_loss, save_path / "losses.json")
