##########################################
#           RAW DATA CONFIGURATION       #
##########################################

# Input directories for raw EM images and masks
raw_image_dir = "/Users/gloof/Desktop/data/cellmap_2d_training_data_nuc/2D_em_masks_100725_6_75nm/em_2d"
raw_mask_dir = "/Users/gloof/Desktop/data/cellmap_2d_training_data_nuc/2D_em_masks_100725_6_75nm/gt_2d"

# Output directory for split dataset
split_output_dir = "tmp"

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
cropped_output_dir = "/Users/gloof/Desktop/code/em_nuclear_segmentation/src/em_nuclear_segmentation/data/preprocessed"

# Plotting and preview settings
plot_path = "preprocessed/size_distribution.png"
preview_grid_path = "preprocessed/preview_grid.png"
max_preview_patches = 25


##########################################
#        TRAINING DATA DIRECTORIES       #
##########################################

# Directories for training and validation data
train_image_dir = "/Users/gloof/Desktop/code/em_nuclear_segmentation/src/em_nuclear_segmentation/data/train/images"
train_mask_dir  = "/Users/gloof/Desktop/code/em_nuclear_segmentation/src/em_nuclear_segmentation/data/train/masks"

val_image_dir   = "/Users/gloof/Desktop/code/em_nuclear_segmentation/src/em_nuclear_segmentation/data/val/images"
val_mask_dir    = "/Users/gloof/Desktop/code/em_nuclear_segmentation/src/em_nuclear_segmentation/data/val/masks"


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
num_epochs = 100
dropout_prob = 0.2  # or 0.0 to disable

# Data augmentation
use_augmentation = True

# Early stopping
use_early_stopping = True
early_stopping_patience = 20  # Stop if val_loss doesn't improve after X epochs

# final model output name
train_output_dir = "test"
model_output_name = "test.pth"
 
##########################################
#            EVALUATION SETTINGS         #
##########################################

# Path to trained model for evaluation
evaluation_model_path = "/Users/gloof/Desktop/code/em_nuclear_segmentation/src/em_nuclear_segmentation/data/best_nuclei_unet_140725.pth"

# Dataset to evaluate (usually test or val)
test_image_dir = "/Users/gloof/Desktop/code/em_nuclear_segmentation/src/em_nuclear_segmentation/data/test/images"
test_mask_dir  = "/Users/gloof/Desktop/code/em_nuclear_segmentation/src/em_nuclear_segmentation/data/test/masks"

# Output paths for evaluation results
evaluation_output_csv = "/Users/gloof/Desktop/code/em_nuclear_segmentation/src/em_nuclear_segmentation/data/results_150725/evaluation_metrics.csv"
prediction_output_dir = "/Users/gloof/Desktop/code/em_nuclear_segmentation/src/em_nuclear_segmentation/data/results_150725/predictions"
save_predictions = True
save_visualizations = True
architecture_diagram_path = "assets/unet_architecture.png"


##########################################
#            PREDICTION SETTINGS         #
##########################################
model_path = "/Users/gloof/Desktop/code/em_nuclear_segmentation/src/em_nuclear_segmentation/data/best_nuclei_unet_140725.pth"
prediction_output_dir = "/Users/gloof/Desktop/code/em_nuclear_segmentation/src/em_nuclear_segmentation/data/results_150725/Matt/predictions"
# Toggle saving of visualization overlays (input vs. prediction)
save_visual_overlay = True
use_histogram_matching = True
histogram_reference_image = "/Users/gloof/Desktop/code/em_nuclear_segmentation/src/em_nuclear_segmentation/data/train/images/image_90.tif" #"path/to/reference_image.tif"



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
