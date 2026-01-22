import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def extract_steer_value(folder_name: str) -> float:
    """Extracts the steer value from a folder name like steps=20_guide=5.0_steer=0.5_type=default."""
    match = re.search(r"steer=([0-9]*\.?[0-9]+)", folder_name)
    return float(match.group(1)) if match else float("inf")

def plot_all_param_images(base_folder: str, save_dir: str = "plots_output"):
    """
    For each unique filename across param folders, plot horizontally across all
    steer values and save the plot.

    Args:
        base_folder (str): Path to the experiments folder (e.g. 'metal_experiments').
        save_dir (str): Folder where plots will be saved.

    Returns:
        List[str]: Filenames of saved plots.
    """
    os.makedirs(save_dir, exist_ok=True)

    # list all param subfolders
    param_folders = [os.path.join(base_folder, f) for f in os.listdir(base_folder)
                     if os.path.isdir(os.path.join(base_folder, f))]

    if not param_folders:
        raise ValueError("No param folders found in base folder.")

    # sort by steer value
    param_folders.sort(key=lambda x: extract_steer_value(os.path.basename(x)))

    # collect all unique filenames from the first folder
    sample_files = [f for f in os.listdir(param_folders[0]) if f.lower().endswith((".jpg", ".png"))]
    if not sample_files:
        raise ValueError("No images found in param folders.")

    saved_plots = []

    # loop through all unique filenames
    for chosen_file in sample_files:
        images = []
        labels = []
        for pf in param_folders:
            img_path = os.path.join(pf, chosen_file)
            if os.path.exists(img_path):
                images.append(mpimg.imread(img_path))
                labels.append(os.path.basename(pf))

        if not images:
            continue

        # plot horizontally
        n = len(images)
        plt.figure(figsize=(4*n, 4))
        for i, (img, label) in enumerate(zip(images, labels)):
            plt.subplot(1, n, i+1)
            plt.imshow(img)
            plt.axis("off")
            steer_val = extract_steer_value(label)
            plt.title(f"$\\alpha$={steer_val}", fontsize=10)  # alpha symbol

        # add main title = filename
        # plt.suptitle(chosen_file, fontsize=12, y=1.02)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # prevent clipping

        # save figure
        save_filename = f"{os.path.splitext(chosen_file)[0]}_comparison.png"
        save_path = os.path.join(save_dir, save_filename)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        saved_plots.append(save_filename)

    print(f"All plots saved in: {os.path.abspath(save_dir)}")
    return saved_plots

def list_saved_plots(save_dir: str = "plots_output"):
    """List all saved plots in the output directory."""
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"{save_dir} not found.")
    plots = [f for f in os.listdir(save_dir) if f.endswith("_comparison.png")]
    print("Available plots:", plots)
    return plots

def load_saved_plot(filename: str, save_dir: str = "plots_output"):
    """
    Load and display a saved plot by filename.

    Args:
        filename (str): Filename of saved plot (e.g. '00001_comparison.png').
        save_dir (str): Directory where plots were saved.
    """
    file_path = os.path.join(save_dir, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    img = mpimg.imread(file_path)
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

