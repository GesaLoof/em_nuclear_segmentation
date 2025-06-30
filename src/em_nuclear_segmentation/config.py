# Input directories for raw dataset
raw_image_dir = "raw_data/images"
raw_mask_dir = "raw_data/masks"

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

