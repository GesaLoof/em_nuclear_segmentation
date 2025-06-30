DATA_PARAMS = {
    "data_path":"path/to/data",
    "binarize":False,
    "target_size":[64, 512, 512],
    "patch_size":[32, 256, 256],
    "augmentations":True
}  

FINE_TUNING = {
    "upload_model_path":"path/to/data",
    "old_steps":2,
}


TRAINING_PARAMS = {
    "loss":"bce",
    "learning_rate": 1e-4,
    "batch_size": 8,
    "epochs": 200,
    "save_model_path":"path/to/data",
    "fine_tuning":True,
    "save_each":True,
}

TEST_PARAMS = {
    "data_path":"path/to/data",
    "binarize":False,
    "target_size":[64, 512, 512],
    "patch_size":[32, 512, 512],
    "batch_size": 2,
    "load_model_path":"path/to/data",
    "load_csv_path":"path/to/data"
}

PRED_PARAMS = {
    "data_path":"path/to/data",
    "final_load_model_path": "path/to/data",
    "batch_size": 2,
    "save_pred_path":"path/to/data"
}