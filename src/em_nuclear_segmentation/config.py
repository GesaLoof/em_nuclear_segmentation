##########################################
#           RAW DATA CONFIGURATION       #
##########################################

# Input directories for raw EM images and masks
raw_image_dir = "path/to/raw/em_images"
raw_mask_dir = "path/to/raw/masks"

# Output directory for split dataset
split_output_dir = "path/to/output/folder"

# Train/val/test split ratios
train_split = 0.7
val_split = 0.15
test_split = 0.15

# Random seed for reproducibility
random_seed = 42


##########################################
#           PATCH CROPPING SETTINGS      #
##########################################

# Cropping parameters
patch_size = 256
edge_mode = "pad"                # Options: "keep", "pad", "overlap"
pad_fill_mode = "mean"           # Options: "zero", "mean", "median", "value:128"

# Output directory for cropped patches
cropped_output_dir = "path/to/cropped/patches/output"

# Plotting and preview settings
plot_path = "preprocessed/size_distribution.png"
preview_grid_path = "preprocessed/preview_grid.png"
max_preview_patches = 25


##########################################
#        TRAINING DATA DIRECTORIES       #
##########################################

# Directories for training and validation data
train_image_dir = "/data/train/images"
train_mask_dir  = "/data/train/masks"

val_image_dir   = "/data/val/images"
val_mask_dir    = "/data/val/masks"


##########################################
#            TRAINING SETTINGS           #
##########################################

# Input image resizing
resize_height = 256
resize_width = 256

# Model I/O channels
in_channels = 1
out_channels = 1

# Training hyperparameters
batch_size = 8
learning_rate = 1e-4
weight_decay = 1e-5
num_epochs = 5
dropout_prob = 0.3  # or 0.0 to disable

# Data augmentation
use_augmentation = True
augmentation_strength = "strong"  # Options: "weak", "medium", "strong"

# Early stopping
use_early_stopping = True
early_stopping_patience = 20  # Stop if val_loss doesn't improve after X epochs

# final model output name
train_output_dir = "training_output_date"
model_output_name = "model.pth"

#resume training from a checkpoint
resume_training = False
resume_checkpoint_path = "data/best_model.pth"  # Path to resume training from
 
##########################################
#            EVALUATION SETTINGS         #
##########################################

# Path to trained model for evaluation
evaluation_model_path = "path/to/model.pth"

# Dataset to evaluate (usually test or val)
test_image_dir = "/data/test/images"
test_mask_dir  = "/data/test/masks"

# Output paths for evaluation results
evaluation_output_csv = "path/to/evaluation_metrics.csv"
prediction_output_dir = "path/to/predictions"
save_predictions = True
save_visualizations = True
architecture_diagram_path = "assets/unet_architecture.png"


##########################################
#            PREDICTION SETTINGS         #
##########################################
model_path = "path/to/model.pth"
prediction_output_dir = "path/to/predictions_output"
save_visual_overlay = True # Toggle saving of visualization overlays (input vs. prediction)
use_histogram_matching = True # whether to apply histogram matching during preprocessing for predictions
histogram_reference_image = "path/to/reference/image.tif" #"path/to/reference_image.tif"
prediction_threshold = 0.1  # Threshold for binarizing predictions


##########################################
#            FINE TUNE SETTINGS         #
##########################################
fine_tune = True
fine_tune_from = "data/best_model.pth"   # path to the pretrained model
freeze_encoder = True                    # whether to freeze encoder
fine_tune_epochs = 20                    # how many epochs to fine-tune
fine_tune_output_dir = "data/"  # output path for fine-tuned model
fine_tune_output_name = "my_finetuned_model.pth"
rescale_uint16_to_uint8 = True  # If True, uint16 images are scaled to match uint8 input range
