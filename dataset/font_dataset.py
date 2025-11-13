import os
import random
from typing import Dict

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def get_nonorm_transform(resolution):
    nonorm_transform =  transforms.Compose(
            [transforms.Resize((resolution, resolution), 
                               interpolation=transforms.InterpolationMode.BILINEAR), 
             transforms.ToTensor()])
    return nonorm_transform


class FontDataset(Dataset):
    """The dataset of font generation  
    """
    def __init__(self, args, phase, transforms=None, scr=False):
        super().__init__()
        self.args = args
        self.root = args.data_root
        self.phase = phase
        self.scr = scr
        if self.scr:
            self.num_neg = args.num_neg

        self.enable_structure = getattr(args, "enable_structure_guidance", False)
        self.structure_cache_root = getattr(args, "structure_cache_root", None)
        feature_keys = getattr(args, "structure_feature_keys", "") or ""
        if isinstance(feature_keys, str):
            self.structure_feature_keys = [key.strip() for key in feature_keys.split(',') if key.strip()]
        else:
            self.structure_feature_keys = [str(key).strip() for key in feature_keys if str(key).strip()]
        self.resolution = args.resolution
        
        # Get Data path
        self.get_path()
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

    def get_path(self):
        self.target_images = []
        # images with related style  
        self.style_to_images = {}
        target_image_dir = f"{self.root}/{self.phase}/TargetImage"
        for style in os.listdir(target_image_dir):
            images_related_style = []
            for img in os.listdir(f"{target_image_dir}/{style}"):
                img_path = f"{target_image_dir}/{style}/{img}"
                self.target_images.append(img_path)
                images_related_style.append(img_path)
            self.style_to_images[style] = images_related_style

    def __getitem__(self, index):
        target_image_path = self.target_images[index]
        target_image_name = target_image_path.split('/')[-1]
        style, content = target_image_name.split('.')[0].split('+')
        
        # Read content image
        content_image_path = f"{self.root}/{self.phase}/ContentImage/{content}.jpg"
        content_image = Image.open(content_image_path).convert('RGB')

        # Random sample used for style image
        images_related_style = self.style_to_images[style].copy()
        images_related_style.remove(target_image_path)
        style_image_path = random.choice(images_related_style)
        style_image = Image.open(style_image_path).convert("RGB")
        
        # Read target image
        target_image = Image.open(target_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)

        if self.transforms is not None:
            content_image = self.transforms[0](content_image)
            style_image = self.transforms[1](style_image)
            target_image = self.transforms[2](target_image)
        
        sample = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image}

        if self.enable_structure and self.structure_feature_keys:
            structure_features = self._load_structure_features(style=style, content=content)
            sample.update(structure_features)
        
        if self.scr:
            # Get neg image from the different style of the same content
            style_list = list(self.style_to_images.keys())
            style_index = style_list.index(style)
            style_list.pop(style_index)
            choose_neg_names = []
            for i in range(self.num_neg):
                choose_style = random.choice(style_list)
                choose_index = style_list.index(choose_style)
                style_list.pop(choose_index)
                choose_neg_name = f"{self.root}/train/TargetImage/{choose_style}/{choose_style}+{content}.jpg"
                choose_neg_names.append(choose_neg_name)

            # Load neg_images
            for i, neg_name in enumerate(choose_neg_names):
                neg_image = Image.open(neg_name).convert("RGB")
                if self.transforms is not None:
                    neg_image = self.transforms[2](neg_image)
                if i == 0:
                    neg_images = neg_image[None, :, :, :]
                else:
                    neg_images = torch.cat([neg_images, neg_image[None, :, :, :]], dim=0)
            sample["neg_images"] = neg_images

        return sample

    def __len__(self):
        return len(self.target_images)

    def _load_structure_features(self, style: str, content: str) -> Dict[str, torch.Tensor]:
        """Load cached structure cues for the given style-content pair.

        Each cached file is expected to be an ``npz`` archive that stores
        arrays named according to ``self.structure_feature_keys``. When a
        feature is missing we fall back to a zero tensor so that downstream
        modules can rely on the key to exist without additional guards.
        """

        features: Dict[str, torch.Tensor] = {}
        cache_available = self.structure_cache_root is not None
        cache_name = f"{style}+{content}.npz"
        cache_path = None
        if cache_available:
            cache_path = os.path.join(self.structure_cache_root, cache_name)

        cache_data = None
        if cache_available and os.path.exists(cache_path):
            try:
                cache_data = np.load(cache_path)
            except ValueError:
                cache_data = None

        for key in self.structure_feature_keys:
            if cache_data is not None and key in cache_data:
                array = cache_data[key]
            else:
                array = np.zeros((self.resolution, self.resolution), dtype=np.float32)

            tensor = torch.from_numpy(array).float()
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim > 3:
                tensor = tensor.reshape(tensor.shape[0], tensor.shape[-2], tensor.shape[-1])

            if tensor.shape[-1] != self.resolution or tensor.shape[-2] != self.resolution:
                tensor = F.interpolate(
                    tensor.unsqueeze(0),
                    size=(self.resolution, self.resolution),
                    mode="bilinear",
                    align_corners=False).squeeze(0)

            features[key] = tensor

        return features
