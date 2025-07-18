import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from em_nuclear_segmentation import config
from em_nuclear_segmentation.utils.transform import (
    get_transforms,
    get_training_augmentation_description
)

def export_training_summary(pdf_path="training_summary.pdf"):
    """
    Generates a multi-page PDF summary:
    - Page 1: Configuration and paths
    - Page 2: Loss curves from training_log.csv
    - Page 3: Data augmentation pipeline and optional architecture image
    """
    with PdfPages(pdf_path) as pdf:

        # === PAGE 1: CONFIG SUMMARY ===
        lines = []
        lines.append("Nuclear Segmentation Training Summary")
        lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Data
        lines.append("### Data Directories")
        lines.append(f"Train images: {config.train_image_dir}")
        lines.append(f"Train masks:  {config.train_mask_dir}")
        lines.append(f"Val images:   {config.val_image_dir}")
        lines.append(f"Val masks:    {config.val_mask_dir}")
        lines.append("")

        # Model
        lines.append("### Model Configuration")
        lines.append(f"Model output name: {config.model_output_name}")
        lines.append(f"Input channels:    {config.in_channels}")
        lines.append(f"Output channels:   {config.out_channels}")
        lines.append("")

        # Training
        lines.append("### Training Parameters")
        lines.append(f"Batch size:         {config.batch_size}")
        lines.append(f"Learning rate:      {config.learning_rate}")
        lines.append(f"Num epochs:         {config.num_epochs}")
        lines.append(f"Early stopping:     {config.use_early_stopping}")
        if config.use_early_stopping:
            lines.append(f"Patience:           {config.early_stopping_patience}")
        lines.append("")

        # Preprocessing
        lines.append("### Preprocessing")
        lines.append(f"Resize (HxW):            {config.resize_height} x {config.resize_width}")
        lines.append(f"Rescale uint16 to uint8: {config.rescale_uint16_to_uint8}")
        lines.append("")

        # Output
        lines.append("### Output")
        lines.append(f"Model path: {os.path.join(config.split_output_dir, config.model_output_name)}")
        lines.append(f"Log file:   {os.path.join(config.split_output_dir, 'training_log.csv')}")
        lines.append("")

        # Save config summary page
        fig1, ax1 = plt.subplots(figsize=(8.5, 11))
        ax1.axis('off')
        ax1.text(0, 1, "\n".join(lines), fontsize=10, verticalalignment='top', family='monospace')
        pdf.savefig(fig1)
        plt.close(fig1)

        # === PAGE 2: TRAINING LOSS CURVE ===
        log_path = os.path.join(config.split_output_dir, "training_log.csv")
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            fig2, ax2 = plt.subplots(figsize=(8.5, 6))
            ax2.plot(df["epoch"], df["train_loss"], label="Training Loss", marker='o')
            ax2.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker='o')
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")
            ax2.set_title("Training and Validation Loss Curve")
            ax2.grid(True)
            ax2.legend()
            pdf.savefig(fig2)
            plt.close(fig2)
        else:
            print(f"Warning: training log not found at: {log_path}. Loss plot skipped.")

        # === PAGE 3: AUGMENTATION + ARCHITECTURE THUMBNAIL ===
        get_transforms()  # populates _applied_augmentations
        aug_lines = get_training_augmentation_description()

        fig3, ax3 = plt.subplots(figsize=(8.5, 11))
        ax3.axis('off')

        text_y = 1.0
        if isinstance(aug_lines, list):
            aug_block = ["Data Augmentation Pipeline:"]
            aug_block.extend(["  • " + line for line in aug_lines])
        else:
            aug_block = ["Data Augmentation Pipeline:", aug_lines]

        ax3.text(0, text_y, "\n".join(aug_block), fontsize=10, verticalalignment='top', family='monospace')

        # Optional: include architecture image thumbnail
        if hasattr(config, "architecture_diagram_path") and os.path.exists(config.architecture_diagram_path):
            try:
                img = plt.imread(config.architecture_diagram_path)
                img_y = text_y - 0.8
                ax3.imshow(img, extent=(0.05, 0.95, -0.5, 0.3), aspect='auto')
            except Exception as e:
                print("Failed to load architecture image:", e)
        else:
            print("Architecture diagram not found or not set in config.")

        pdf.savefig(fig3)
        plt.close(fig3)

    print(f"PDF training summary exported to {pdf_path}")

if __name__ == "__main__":
    export_training_summary()
