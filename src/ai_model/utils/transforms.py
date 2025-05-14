import albumentations as A
from albumentations.pytorch import ToTensorV2
from ai_model.config import AUGMENTATION_CONFIG

def get_transforms(phase):
    """Get data augmentation transforms for the specified phase."""
    config = AUGMENTATION_CONFIG[phase]
    
    if phase == "train":
        return A.Compose([
            A.RandomResizedCrop(
                size=(config["image_size"], config["image_size"]),
                scale=config["scale"]
            ),
            A.HorizontalFlip(p=config["horizontal_flip_prob"]),
            A.VerticalFlip(p=config["vertical_flip_prob"]),
            A.RandomRotate90(p=config["rotate_prob"]),
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ], p=config["brightness_contrast_prob"]),
            A.OneOf([
                A.ElasticTransform(
                    alpha=120,
                    sigma=120 * 0.05,
                    alpha_affine=120 * 0.03,
                    p=1
                ),
                A.GridDistortion(p=1),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
            ], p=config["elastic_transform_prob"]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(
                height=config["image_size"],
                width=config["image_size"]
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ]) 