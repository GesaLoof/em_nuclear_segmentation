# Input directories for raw dataset
raw_image_dir = "/Users/gloof/Desktop/data/cellmap_2d_training_data_nuc/em_2D"
raw_mask_dir = "/Users/gloof/Desktop/data/cellmap_2d_training_data_nuc/gt_2D"

# Output directory for split dataset
split_output_dir = "data"

# Split ratios
train_split = 0.7
val_split = 0.15
test_split = 0.15

# Random seed for reproducibility
random_seed = 42

# Data paths
train_image_dir = "data/train/images"
train_mask_dir = "data/train/masks"

# Validation directories
val_image_dir = "data/val/images"
val_mask_dir = "data/val/masks"

# Training hyperparameters
batch_size = 4
learning_rate = 1e-4
num_epochs = 10

# Image settings
resize_height = 256
resize_width = 256

# Augmentation
use_augmentation = True

# Model
in_channels = 1
out_channels = 1

# Checkpoint
checkpoint_path = "unet_nuclei.pth"

# Output directory for predictions
prediction_output_dir = "predictions"

# Log path and final model path
checkpoint_path = "unet_nuclei.pth"
split_output_dir = "data"

