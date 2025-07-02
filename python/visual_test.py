import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re


def complex_array_to_abs(array):
    density = np.sum(array**2, axis=2)
    return density

def find_files(directory):
    # Regular expression patterns
    pattern_single = re.compile(r'^final_(\d+)\.npy$')
    pattern_double = re.compile(r'^partial_(\d+)_(\d+)\.npy$')

    # Store results
    results = {}

    # Convert directory to Path object
    directory = Path(directory)

    # Find all files matching the single pattern
    single_files = list(directory.glob('final_*.npy'))

    # Process single files
    for file_path in single_files:
        filename = file_path.name
        single_match = pattern_single.match(filename)
        if single_match:
            x = single_match.group(1)
            if x not in results:
                results[x] = {'single': [], 'doubles': []}
            results[x]['single'].append(file_path)
    
    # Find all files matching the double pattern
    double_files = list(directory.glob('partial_*_*.npy'))

    # Process double files
    for file_path in double_files:
        filename = file_path.name
        double_match = pattern_double.match(filename)
        if double_match:
            x, y = double_match.groups()
            if x in results:
                results[x]['doubles'].append((int(y), file_path))
    
    return results

def visualize_arrays(x, single_files, double_files):
    # Load single files and double files as numpy arrays
    single_arrays = [np.load(file).astype(np.float32) for file in single_files]
    double_arrays = [(y, np.load(file).astype(np.float32)) for y, file in sorted(double_files)]

    # Create a figure with subplots
    num_single = len(single_arrays)
    num_double = len(double_arrays)
    num_plots = num_single + num_double
    
    fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))
    if num_plots == 1:
        axes = [axes]  # Ensure axes is always iterable


    # Plot double arrays
    for j, (y, array) in enumerate(double_arrays):
        axes[j].imshow(complex_array_to_abs(array))
        axes[j].set_title(f'{10 * y}%')
        axes[j].axis('off')

    # Plot single arrays
    for i, array in enumerate(single_arrays):
        axes[num_double+i].imshow(complex_array_to_abs(array))
        axes[num_double+i].set_title('Final')
        axes[num_double+i].axis('off')

    plt.suptitle(f'Visualizations for example {x}')
    plt.tight_layout()
    plt.savefig("figure1.png")

def visualize_arrays_sep(x, single_files, double_files):
    # Load single files and double files as numpy arrays
    single_arrays = [np.load(file).astype(np.float32) for file in single_files]
    double_arrays = [(y, np.load(file).astype(np.float32)) for y, file in sorted(double_files)]

    # Create a figure with subplots
    num_single = len(single_arrays)
    num_double = len(double_arrays)
    num_plots = num_single + num_double
    
    fig, axes = plt.subplots(2, num_plots, figsize=(15, 5))
    if num_plots == 1:
        axes = [axes]  # Ensure axes is always iterable


    # Plot double arrays
    for j, (y, array) in enumerate(double_arrays):
        axes[0, j].imshow(array[:,:,0])
        axes[0, j].set_title(f'Re {10 * y}%')
        axes[0, j].axis('off')

        axes[1, j].imshow(array[:,:,1])
        axes[1, j].set_title(f'Im {10 * y}%')
        axes[1, j].axis('off')

    # Plot single arrays
    for i, array in enumerate(single_arrays):
        axes[0, num_double+i].imshow(array[:,:,0])
        axes[0, num_double+i].set_title('Re 100%')
        axes[0, num_double+i].axis('off')

        axes[1, num_double+i].imshow(array[:,:,1])
        axes[1, num_double+i].set_title('Im 100%')
        axes[1, num_double+i].axis('off')

    plt.suptitle(f'Visualizations for example {x}')
    plt.tight_layout()
    plt.savefig("figure1.png")

def main():
    directory = 'dftk/'
    results = find_files(directory)
    
    for x in results:
        single_files = results[x]['single']
        double_files = results[x]['doubles']
        visualize_arrays_sep(x, single_files, double_files)

if __name__ == '__main__':
    main()
