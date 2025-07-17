import albumentations as A
from albumentations.pytorch import ToTensorV2
from em_nuclear_segmentation import config

# Shared reference for augmentations applied
_applied_augmentations = {}

def get_transforms():
    """
    Returns composed Albumentations transform for training.
    Stores applied transform parameters in _applied_augmentations.
    """
    global _applied_augmentations
    _applied_augmentations = {}

    transforms = [A.Resize(config.resize_height, config.resize_width)]
    _applied_augmentations["Resize"] = {
        "height": config.resize_height,
        "width": config.resize_width
    }

    if config.use_augmentation:
        # Geometric
        _applied_augmentations["HorizontalFlip"] = {"p": 0.5}
        transforms.append(A.HorizontalFlip(p=0.5))

        _applied_augmentations["VerticalFlip"] = {"p": 0.5}
        transforms.append(A.VerticalFlip(p=0.5))

        _applied_augmentations["RandomRotate90"] = {"p": 0.5}
        transforms.append(A.RandomRotate90(p=0.5))

        _applied_augmentations["Affine"] = {
            "translate_percent": 0.02,
            "scale": (0.95, 1.05),
            "rotate": (-10, 10),
            "shear": (-5, 5),
            "mode": "reflect",
            "p": 0.3
        }
        transforms.append(
            A.Affine(
                translate_percent=0.02,
                scale=(0.95, 1.05),
                rotate=(-10, 10),
                shear=(-5, 5),
                mode=1,  # reflect
                p=0.3
            )
        )

        # Appearance
        _applied_augmentations["RandomBrightnessContrast"] = {
            "brightness_limit": 0.1,
            "contrast_limit": 0.1,
            "p": 0.4
        }
        transforms.append(A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4))

        _applied_augmentations["GaussNoise"] = {"var_limit": (5.0, 20.0), "p": 0.3}
        transforms.append(A.GaussNoise(var_limit=(5.0, 20.0), p=0.3))

        _applied_augmentations["MotionBlur"] = {"blur_limit": 3, "p": 0.1}
        transforms.append(A.MotionBlur(blur_limit=3, p=0.1))

        _applied_augmentations["ElasticTransform"] = {
            "alpha": 0.5,
            "sigma": 10,
            "alpha_affine": 5,
            "border_mode": "reflect",
            "p": 0.1
        }
        transforms.append(A.ElasticTransform(alpha=0.5, sigma=10, alpha_affine=5, border_mode=1, p=0.1))

    # Final normalization and tensor conversion
    transforms.extend([
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])
    _applied_augmentations["Normalize"] = {"mean": 0.5, "std": 0.5}
    _applied_augmentations["ToTensorV2"] = {}

    return A.Compose(transforms)

def get_val_transforms():
    return A.Compose([
        A.Resize(config.resize_height, config.resize_width),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

def get_training_augmentation_description():
    """
    Dynamically describes the applied training augmentation pipeline.
    Reads parameters from _applied_augmentations generated during get_transforms().
    """
    if not config.use_augmentation:
        return "No data augmentation applied."

    lines = []
    for key, params in _applied_augmentations.items():
        if not params:
            lines.append(f"{key}")
        else:
            arg_str = ", ".join(f"{k}={v}" for k, v in params.items())
            lines.append(f"{key}({arg_str})")
    return lines
