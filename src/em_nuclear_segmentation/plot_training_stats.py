import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_training_stats(log_path, output_path="training_curve.png"):
    """
    Plot training and validation loss curves from training_log.csv.

    Parameters:
        log_path (str): Path to training_log.csv
        output_path (str): Path to save the output PNG plot
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found at: {log_path}")

    # Load the CSV into a DataFrame
    df = pd.read_csv(log_path)

    # Check for expected columns
    if not {"epoch", "train_loss", "val_loss"}.issubset(df.columns):
        raise ValueError("CSV must contain 'epoch', 'train_loss', and 'val_loss' columns.")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Training Loss", marker='o')
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Training statistics plot saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training and validation loss curves from log.")
    parser.add_argument("--log", type=str, default="data/training_log.csv", help="Path to training_log.csv")
    parser.add_argument("--output", type=str, default="training_curve.png", help="Output image path")
    args = parser.parse_args()

    plot_training_stats(args.log, args.output)
