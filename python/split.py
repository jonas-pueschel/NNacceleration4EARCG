import random
import shutil
import os
from pathlib import Path

# Set the paths to your files and folders
os.chdir("./data/")

source_folder = Path("./")
train_folder = Path("./training")
eval_folder = Path("./evaluation")

# decide if split the runs 
split_runs = False
n_samples = 20

# Set the percentage of data to be used for evaluation
evaluation_percentage = 10

# Create train and evaluation folders if they don't exist
Path.mkdir(train_folder, parents=True, exist_ok=True)
Path.mkdir(eval_folder, parents=True, exist_ok=True)


if split_runs:
    # Get a list of all files in the source folder
    all_files = list(source_folder.glob("final*.npy"))

    # Calculate the number of files for evaluation
    num_eval_files = int(len(all_files) * (evaluation_percentage / 100))

    # Randomly select files for evaluation
    eval_files = random.sample(all_files, num_eval_files)

    # Move files to the appropriate folders
    for file_name in all_files:
        file_name_str = str(file_name).split(".npy")[0]
        file_path = source_folder / file_name
        num = file_name_str.split("_")[1]
        print(num)


        if file_name in eval_files:
            shutil.move(file_path, eval_folder / file_name)
        else:
            shutil.move(file_path, train_folder / file_name)


        for percentage in range(1,n_samples + 1):
            partial_name = f"partial_{num}_{percentage}.npy"
            grad_name = f"grad_{num}_{percentage}.npy"
            partial_file_path = source_folder / partial_name
            grad_file_name = source_folder / grad_name

            if file_name in eval_files:
                shutil.move(grad_file_name, eval_folder / grad_name)
                shutil.move(partial_file_path, eval_folder / partial_name)
            else:
                shutil.move(grad_file_name, train_folder / grad_name)
                shutil.move(partial_file_path, train_folder / partial_name)

else:
    # Get a list of all files in the source folder
    all_files = list(source_folder.glob("partial*.npy"))



    # Calculate the number of files for evaluation
    num_eval_files = int(len(all_files) * (evaluation_percentage / 100))

    # Randomly select files for evaluation
    eval_files = random.sample(all_files, num_eval_files)

    # Move files to the appropriate folders
    for file_name in all_files:
        file_name_str = str(file_name)
        file_path = source_folder / file_name
        num = file_name_str.split("_")[1]
        percentage = file_name_str.split("_")[2].split(".")[0]
        final_name = f"final_{num}.npy"
        grad_name = f"grad_{num}_{percentage}.npy"
        final_file_path = source_folder / final_name
        grad_file_name = source_folder / grad_name

        if file_name in eval_files:
            shutil.move(file_path, eval_folder / file_name)
            shutil.move(grad_file_name, eval_folder / grad_name)
            shutil.copy(final_file_path, eval_folder / final_name)
        else:
            shutil.move(file_path, train_folder / file_name)
            shutil.move(grad_file_name, train_folder / grad_name)
            shutil.copy(final_file_path, train_folder / final_name)
