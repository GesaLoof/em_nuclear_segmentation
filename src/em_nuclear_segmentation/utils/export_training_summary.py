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

def _safe_plot(ax, df, xcol, ycol, label):
    if ycol in df.columns:
        ax.plot(df[xcol], df[ycol], marker='o', label=label)
        return True
    else:
        print(f"[export_training_summary] Column '{ycol}' not found in training_log.csv; skipping.")
        return False

def export_training_summary(pdf_path=os.path.join(config.train_output_dir,"training_summary.pdf")):
    """
    Generates a multi-page PDF summary:
    - Page 1: Configuration and paths
    - Page 2: Loss curves from training_log.csv
    - Page 3: Validation metrics (Accuracy, Dice, IoU) — also plots training metrics if present
    - Page 4: Data augmentation pipeline and optional architecture image
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
        lines.append(f"Weight decay:      {config.weight_decay}")
        lines.append(f"Dropout probability:      {config.dropout_prob}")
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
        lines.append(f"Model path: {os.path.join(config.train_output_dir, config.model_output_name)}")
        lines.append(f"Log file:   {os.path.join(config.train_output_dir, 'training_log.csv')}")
        lines.append("")

        fig1, ax1 = plt.subplots(figsize=(8.5, 11))
        ax1.axis('off')
        ax1.text(0, 1, "\n".join(lines), fontsize=10, verticalalignment='top', family='monospace')
        pdf.savefig(fig1)
        plt.close(fig1)

        # === PAGE 2: TRAINING & VALIDATION LOSS CURVE ===
        log_path = os.path.join(config.train_output_dir, "training_log.csv")
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)

            # ensure expected columns exist
            if "epoch" not in df.columns:
                raise ValueError("training_log.csv must contain an 'epoch' column.")

            fig2, ax2 = plt.subplots(figsize=(8.5, 6))
            had_any = False
            had_any |= _safe_plot(ax2, df, "epoch", "train_loss", "Training Loss")
            had_any |= _safe_plot(ax2, df, "epoch", "val_loss", "Validation Loss")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")
            ax2.set_title("Training and Validation Loss Curve")
            ax2.grid(True)
            if had_any:
                ax2.legend()
            pdf.savefig(fig2)
            plt.close(fig2)
        else:
            print(f"Warning: training log not found at: {log_path}. Loss plot skipped.")

        # === PAGE 3: ACCURACY-RELATED METRICS (VAL + optional TRAIN) ===
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)

            # You may have logged only validation metrics:
            # expected validation columns: val_acc, val_dice, val_iou
            # optional training columns:   train_acc, train_dice, train_iou
            metric_pages = [
                ("Accuracy",  "train_acc", "val_acc",  "Accuracy"),
                ("Dice",      "train_dice","val_dice", "Dice Coefficient"),
                ("IoU",       "train_iou", "val_iou",  "Intersection over Union"),
            ]

            for title, train_col, val_col, ylabel in metric_pages:
                figm, axm = plt.subplots(figsize=(8.5, 6))
                had_any = False
                had_any |= _safe_plot(axm, df, "epoch", train_col, f"Training {title}")
                had_any |= _safe_plot(axm, df, "epoch", val_col,   f"Validation {title}")
                axm.set_xlabel("Epoch")
                axm.set_ylabel(ylabel)
                axm.set_title(f"{title} over Epochs")
                axm.grid(True)
                if had_any:
                    axm.legend()
                pdf.savefig(figm)
                plt.close(figm)
        else:
            print(f"Warning: training log not found at: {log_path}. Metrics plots skipped.")

        # === PAGE 4: AUGMENTATION + ARCHITECTURE THUMBNAIL ===
        get_transforms()  # populates _applied_augmentations internally
        aug_lines = get_training_augmentation_description()

        fig3, ax3 = plt.subplots(figsize=(8.5, 11))
        ax3.axis('off')

        text_y = 1.0
        if isinstance(aug_lines, list):
            aug_block = ["Data Augmentation Pipeline:"]
            aug_block.extend([line for line in aug_lines])  # already formatted by your helper
        else:
            aug_block = ["Data Augmentation Pipeline:", str(aug_lines)]

        ax3.text(0, text_y, "\n".join(aug_block), fontsize=10, verticalalignment='top', family='monospace')

        # Optional: include architecture image thumbnail
        if hasattr(config, "architecture_diagram_path") and os.path.exists(config.architecture_diagram_path):
            try:
                img = plt.imread(config.architecture_diagram_path)
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
