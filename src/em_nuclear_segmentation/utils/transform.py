import albumentations as A
from albumentations.pytorch import ToTensorV2
from em_nuclear_segmentation import config

# Shared reference for augmentations applied
_applied_augmentations = {}

def _push_record(name, params):
    # Helper to store what's applied
    global _applied_augmentations
    _applied_augmentations[name] = params

def get_transforms():
    """
    Returns composed Albumentations transform for training.
    Stores applied transform parameters in _applied_augmentations.
    Strength is controlled by config.augmentation_strength: "weak"|"medium"|"strong".
    """
    strength = getattr(config, "augmentation_strength", "medium").lower()
    global _applied_augmentations
    _applied_augmentations = {}

    transforms = [A.Resize(config.resize_height, config.resize_width)]
    _push_record("Resize", {"height": config.resize_height, "width": config.resize_width})

    if getattr(config, "use_augmentation", True):

        if strength == "weak":
            # --- Geometry (light) ---
            _push_record("HorizontalFlip", {"p": 0.5})
            transforms.append(A.HorizontalFlip(p=0.5))

            _push_record("RandomRotate90", {"p": 0.5})
            transforms.append(A.RandomRotate90(p=0.5))

            # Use Affine (fast path); DO NOT use ShiftScaleRotate API here
            _push_record("Affine", {
                "translate_percent": 0.05,
                "scale": (0.95, 1.05),
                "rotate": (-10, 10),
                "shear": (-5, 5),
                "border_mode": 1,
                "p": 0.3
            })
            transforms.append(A.Affine(
                translate_percent=0.05,
                scale=(0.95, 1.05),
                rotate=(-10, 10),
                shear=(-5, 5),
                border_mode=1,
                p=0.3
            ))

            # --- Appearance (light) ---
            _push_record("RandomBrightnessContrast", {"brightness_limit": 0.10, "contrast_limit": 0.10, "p": 0.3})
            transforms.append(A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.3))

            _push_record("GaussNoise", {"std_range": (0.01, 0.04), "mean_range": (-0.02, 0.02), "p": 0.2})
            transforms.append(A.GaussNoise(std_range=(0.01, 0.04), mean_range=(-0.02, 0.02), p=0.2))

            _push_record("MotionBlur", {"blur_limit": 3, "p": 0.1})
            transforms.append(A.MotionBlur(blur_limit=3, p=0.1))

            _push_record("CoarseDropout", {"holes": 2, "max_h_size": 16, "max_w_size": 16, "fill_value": 0, "p": 0.1})
            transforms.append(A.CoarseDropout(holes=2, max_h_size=16, max_w_size=16, fill_value=0, p=0.1))

        elif strength == "strong":
            # --- Geometry (strong but fast) ---
            _push_record("Affine", {
                "translate_percent": 0.15,
                "scale": (0.75, 1.25),
                "rotate": (-30, 30),
                "shear": (-12, 12),
                "border_mode": 1,
                "p": 0.7
            })
            transforms.append(A.Affine(
                translate_percent=0.15,
                scale=(0.75, 1.25),
                rotate=(-30, 30),
                shear=(-12, 12),
                border_mode=1,
                p=0.7
            ))

            # Faster combo: Elastic or Grid (no PiecewiseAffine)
            _push_record("OneOf_ElasticOrGrid", {
                "ElasticTransform": {"alpha": 20, "sigma": 10, "border_mode": 1},
                "GridDistortion": {"num_steps": 5, "distort_limit": 0.4, "border_mode": 1},
                "p": 0.5
            })
            transforms.append(A.OneOf([
                A.ElasticTransform(alpha=20, sigma=10, border_mode=1, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.4, border_mode=1, p=1.0),
            ], p=0.5))

            _push_record("RandomRotate90", {"p": 0.5})
            transforms.append(A.RandomRotate90(p=0.5))

            _push_record("HorizontalFlip", {"p": 0.5})
            transforms.append(A.HorizontalFlip(p=0.5))

            _push_record("VerticalFlip", {"p": 0.5})
            transforms.append(A.VerticalFlip(p=0.5))

            # --- Appearance (strong) ---
            _push_record("OneOf_BrightnessContrastGammaCLAHE", {
                "RandomBrightnessContrast": {"brightness_limit": 0.35, "contrast_limit": 0.35},
                "RandomGamma": {"gamma_limit": (70, 130)},
                "CLAHE": {"clip_limit": 3.0},
                "p": 0.7
            })
            transforms.append(A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=1.0),
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
                A.CLAHE(clip_limit=3.0, p=1.0),
            ], p=0.7))

            # --- Noise / Blur (strong) ---
            _push_record("OneOf_NoiseBlur", {
                "GaussNoise": {"std_range": (0.02, 0.08), "mean_range": (-0.02, 0.02)},
                "MultiplicativeNoise": {"multiplier": (0.9, 1.1)},
                "MotionBlur": {"blur_limit": 7},
                "p": 0.5
            })
            transforms.append(A.OneOf([
                A.GaussNoise(std_range=(0.02, 0.08), mean_range=(-0.02, 0.02), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
            ], p=0.5))

            # --- Occlusion ---
            _push_record("CoarseDropout", {"holes": 6, "max_h_size": 48, "max_w_size": 48, "fill_value": 0, "p": 0.2})
            transforms.append(A.CoarseDropout(holes=6, max_h_size=48, max_w_size=48, fill_value=0, p=0.2))

        else:
            # === "medium" (your original spirit, with corrected APIs and a faster warp combo) ===
            # --- Geometry ---
            _push_record("HorizontalFlip", {"p": 0.5})
            transforms.append(A.HorizontalFlip(p=0.5))

            _push_record("VerticalFlip", {"p": 0.5})
            transforms.append(A.VerticalFlip(p=0.5))

            _push_record("RandomRotate90", {"p": 0.5})
            transforms.append(A.RandomRotate90(p=0.5))

            _push_record("Affine", {
                "translate_percent": 0.0625,
                "scale": (0.9, 1.1),
                "rotate": (-15, 15),
                "shear": (-5, 5),
                "border_mode": 1,
                "p": 0.3
            })
            transforms.append(A.Affine(
                translate_percent=0.0625,
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                shear=(-5, 5),
                border_mode=1,
                p=0.3
            ))

            _push_record("OneOf_ElasticOrGrid", {
                "ElasticTransform": {"alpha": 10, "sigma": 10, "border_mode": 1},
                "GridDistortion": {"num_steps": 5, "distort_limit": 0.3, "border_mode": 1},
                "p": 0.4
            })
            transforms.append(A.OneOf([
                A.ElasticTransform(alpha=10, sigma=10, border_mode=1, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.3, border_mode=1, p=1.0),
            ], p=0.4))

            # --- Appearance ---
            _push_record("RandomBrightnessContrast", {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.4})
            transforms.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4))

            _push_record("GaussNoise", {"std_range": (0.02, 0.06), "mean_range": (-0.02, 0.02), "p": 0.3})
            transforms.append(A.GaussNoise(std_range=(0.02, 0.06), mean_range=(-0.02, 0.02), p=0.3))

            _push_record("MotionBlur", {"blur_limit": 5, "p": 0.2})
            transforms.append(A.MotionBlur(blur_limit=5, p=0.2))

            _push_record("CoarseDropout", {"holes": 4, "max_h_size": 32, "max_w_size": 32, "fill_value": 0, "p": 0.2})
            transforms.append(A.CoarseDropout(holes=4, max_h_size=32, max_w_size=32, fill_value=0, p=0.2))

    # Final normalization and tensor conversion
    transforms.extend([
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])
    _push_record("Normalize", {"mean": 0.5, "std": 0.5})
    _push_record("ToTensorV2", {})

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
    if not getattr(config, "use_augmentation", True):
        return "No data augmentation applied."

    lines = []
    for key, params in _applied_augmentations.items():
        if not params:
            lines.append(f"{key}")
        else:
            arg_str = ", ".join(f"{k}={v}" for k, v in params.items())
            lines.append(f"{key}({arg_str})")
    return lines
