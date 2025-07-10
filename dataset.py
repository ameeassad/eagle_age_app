from pycocotools.coco import COCO
import numpy as np
import os
import ast
import math
import pandas as pd
import json
from IPython.display import display
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
)
import torchvision.transforms.functional as F
from torchvision.transforms.functional import resize, pad
import pytorch_lightning as pl

from data.transforms import SynchTransforms, RGBTransforms, TransformerRGBTransforms, ValTransforms, resize_and_pad, rotate_image
from preprocess.preprocess_utils import create_mask, create_skeleton_channel


class ArtportalenDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for the Artportalen dataset, handles the loading, 
    preprocessing, and transformation of the dataset 
    for training, validation, and testing.

    Args:
        data_dir (str): Path to the dataset directory.
        preprocess_lvl (int): Level of preprocessing to apply to the images.
            0: original image
            1: bounding box cropped image
            2: masked image
            3: masked + pose (skeleton) image in 1 channel
            4: masked + body parts in channels

        batch_size (int): Number of samples per batch.
        size (int): Size of the image for resizing.
        mean (float or tuple): Mean for normalization.
        std (float or tuple): Standard deviation for normalization.
        test (bool): Flag to indicate if in test mode.
        cache_dir (str): Path to the directory for caching preprocessed images.
        use_advanced_aug (bool): Whether to use advanced augmentations for transformer models.
        
    Attributes:
        train_transforms (callable): Transformations applied to the training dataset.
        val_transforms (callable): Transformations applied to the validation dataset.
    """
    def __init__(self, data_dir, preprocess_lvl=0, batch_size=8, size=256, mean=0.5, std=0.5, test=False, cache_dir='dataset/data_cache', use_advanced_aug=False):
        super().__init__()
        self.data_dir = data_dir
        self.preprocess_lvl = preprocess_lvl
        self.batch_size = batch_size
        self.size = size
        self.mean = (mean, mean, mean) if isinstance(mean, float) else tuple(mean)
        self.std = (std, std, std) if isinstance(std, float) else tuple(std)
        self.test = test
        self.cache_dir = cache_dir
        self.use_advanced_aug = use_advanced_aug

        if preprocess_lvl == 3:
            self.skeleton = True
        else:
            self.skeleton = False
        

        # transformations
        if self.skeleton:         
            self.train_transforms = SynchTransforms(mean=self.mean, std=self.std)
            self.val_transforms = ValTransforms(mean=self.mean, std=self.std, skeleton=True)
        else:
            # Use advanced augmentations for transformer models if specified
            if self.use_advanced_aug:
                self.train_transforms = TransformerRGBTransforms(
                    mean=self.mean, 
                    std=self.std,
                    mixup_alpha=0.2,
                    cutmix_alpha=1.0,
                    mixup_prob=0.3,
                    cutmix_prob=0.3,
                    random_erasing_prob=0.3,
                    advanced_aug_prob=0.8
                )
            else:
                self.train_transforms = RGBTransforms(mean=self.mean, std=self.std)
            
            self.val_transforms = Compose([
                # Resize(self.size),
                # Pad((self.size - 1, self.size - 1), padding_mode='constant'),
                ToTensor(),
                Normalize(mean=self.mean, std=self.std)
            ])

    def prepare_data(self):
        """
        Prepares the dataset, downloading or splitting it if needed. 
        In test mode, it prepares the testing data.
        """
        # download, split, etc.
        if self.test:
            self.prepare_testing_data(self.data_dir)

    def prepare_testing_data(self, image_dir="testing/images"):
        """
        Prepares the testing data by creating a COCO-like JSON annotation from the images.
        Inside COCOBuilder, it uses YOLOv8 to detect the bounding boxes and segmentations.
        Finally calls setup_testing().

        Args:
            image_dir (str): Path to the directory containing the test images.
        """
        coco = COCOBuilder(image_dir, testing=True)
        coco.setup_testing()
        coco.fill_coco()
        coco.create_coco_format_json("experiments/testing/coco_training.json")

        self.setup_testing("experiments/testing/coco_training.json")

    def setup_testing(self, test_annot):
        """
        Set up the testing dataset using COCO annotations and convert it to a DataFrame.
        Is called in prepare_testing_data().

        Args:
            test_annot (str): Path to the COCO annotations file for testing.
        """
        # Load COCO annotations
        with open(test_annot, 'r') as f:
            test_data = json.load(f)
        # Initialize COCO objects
        test_coco = COCO(test_annot)
        # Convert annotations to DataFrame
        test_df = self.coco_to_dataframe(test_coco)
        print(f"Test: {len(test_df)}")
        self.num_classes = 5
        print(f"Number of classes: {self.num_classes}")
        self.train_dataset = EagleDataset(test_df, self.data_dir, self.train_transforms, size=self.size)
        self.val_dataset = EagleDataset(test_df, self.data_dir, self.val_transforms, size=self.size, test=self.test)

    def setup_from_csv(self, train_csv, val_csv, stage=None):
        """
        Set up the dataset using CSV files containing the training and validation data.

        Args:
            train_csv (str): Path to the CSV file for training data.
            val_csv (str): Path to the CSV file for validation data.
            stage (str, optional): The stage for which the setup is being done (e.g., 'fit', 'test').
        """
         # Load the CSV files
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        # Ensure 0-indexed labels
        train_df['category_id'] = train_df['category_id'] - train_df['category_id'].min()
        val_df['category_id'] = val_df['category_id'] - val_df['category_id'].min()
        print('Unique train labels:', sorted(train_df['category_id'].unique()))
        print('Unique val labels:', sorted(val_df['category_id'].unique()))
        assert train_df['category_id'].min() == 0, f"Train labels must be 0-indexed, got min {train_df['category_id'].min()}"
        assert val_df['category_id'].min() == 0, f"Val labels must be 0-indexed, got min {val_df['category_id'].min()}"

        # Columns renaming
        column_names = {
            'annot_id': 'id',
            'image_filename': 'file_name',
            'Reporter': 'photographer',
        }

        def jpg_extension(filename):
            filename = str(filename)
            if not filename.lower().endswith('.jpg') or not filename.lower().endswith('.jpeg') or not filename.lower().endswith('.png'):
                return f"{filename}.jpg"
            return filename

        train_df = train_df.rename(columns=column_names)
        train_df['file_name'] = train_df['file_name'].apply(jpg_extension)
        val_df = val_df.rename(columns=column_names)
        val_df['file_name'] = val_df['file_name'].apply(jpg_extension)

        # Print the number of samples in train and validation sets
        print(f"Train: {len(train_df)} Val: {len(val_df)}")

        # Initialize the datasets
        self.train_dataset = EagleDataset(train_df, self.data_dir, self.train_transforms, size=self.size, preprocess_lvl=self.preprocess_lvl, cache_dir=self.cache_dir)
        self.val_dataset = EagleDataset(val_df, self.data_dir, self.val_transforms, size=self.size, preprocess_lvl=self.preprocess_lvl, cache_dir=self.cache_dir)

        # Check the number of unique classes
        unique_classes = train_df['category_id'].unique()
        print(f"Unique classes in dataset: {unique_classes}")
        self.num_classes = len(unique_classes)
        print(f"Number of classes: {self.num_classes}")

    def setup_from_coco(self, train_annot, val_annot, stage=None):
        """
        Set up the dataset using COCO-style annotations for training and validation.

        Args:
            train_annot (str): Path to the COCO annotations file for training.
            val_annot (str): Path to the COCO annotations file for validation.
            stage (str, optional): The stage for which the setup is being done (e.g., 'fit', 'test').
        """
        # Load COCO annotations
        with open(train_annot, 'r') as f:
            train_data = json.load(f)
        with open(val_annot, 'r') as f:
            val_data = json.load(f)

        # Initialize COCO objects
        train_coco = COCO(train_annot)
        val_coco = COCO(val_annot)

        # Convert annotations to DataFrame
        train_df = self.coco_to_dataframe(train_coco)
        val_df = self.coco_to_dataframe(val_coco)

        print(f"Train: {len(train_df)} Val: {len(val_df)}")

        self.train_dataset = EagleDataset(train_df, self.data_dir, self.train_transforms, size=self.size, preprocess_lvl=self.preprocess_lvl, cache_dir=self.cache_dir)
        self.val_dataset = EagleDataset(val_df, self.data_dir, self.val_transforms, size=self.size, preprocess_lvl=self.preprocess_lvl, cache_dir=self.cache_dir)

        # Check number of classes
        unique_classes = train_df['category_id'].unique()
        print(f"Unique classes in dataset: {unique_classes}")
        self.num_classes = len(unique_classes)
        print(f"Number of classes: {self.num_classes}")

    def coco_to_dataframe(self, coco):
        """
        Converts COCO annotations to a pandas DataFrame.

        Args:
            coco (COCO): COCO object containing annotations.

        Returns:
            pd.DataFrame: DataFrame with image and annotation information.
        """
        data = []
        for ann in coco.anns.values():
            img_info = coco.loadImgs(ann['image_id'])[0]

            file_name = img_info['file_name']
            if '.' not in file_name:
                file_name += '.jpg'

            data.append({
                # 'id': ann['id'], # Annotation ID
                'image_id': ann['image_id'],
                'file_name': file_name,
                'height': img_info['height'],
                'width': img_info['width'],
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'area': ann['area'],
                'iscrowd': ann['iscrowd'],
                'segmentation': ann['segmentation'],
                'id': ann['id'],

            })
        return pd.DataFrame(data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
    
    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)


class EagleDataset(Dataset):
    """
    Custom dataset class for loading and preprocessing eagle image data with optional skeleton.

    Args:
        dataframe (pd.DataFrame): DataFrame containing image file paths and annotations.
        data_dir (str): Directory containing the image files.
        transform (callable, optional): Transformations to be applied to the images.
        size (int): Size to which the images should be resized.
        test (bool): Whether this is test data.
        skeleton (bool): Whether to include skeleton channel.
    """
    def __init__(self, dataframe, data_dir, transform=None, size=256, test=False, preprocess_lvl=2, cache_dir='dataset/data_cache'):
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.transform = transform
        self.size = size
        self.test = test
        self.preprocess_lvl = preprocess_lvl
        self.cache_dir = cache_dir

        if preprocess_lvl==3: 
            self.skeleton = True
        else:
            self.skeleton = False

        # Load cache from disk if available
        # self.mask_cache = self.load_cache('mask_cache.pkl') if cache_dir else {}
        self.mask_cache = {}
        self.mask_dir = os.path.join(cache_dir, "masks")
        os.makedirs(self.mask_dir, exist_ok=True)

        if self.skeleton:
            # self.skeleton_transform = skeleton
            self.skeleton_category = AKSkeletonCategory()
            self.skeleton_cache = {}
            # self.skeleton_cache = self.load_cache('skeleton_cache.pkl') if cache_dir and skeleton else {}
            self.skeleton_dir = os.path.join(cache_dir, "skeletons")
            os.makedirs(self.skeleton_dir, exist_ok=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Returns an item (image and label) from the dataset at a given index.

        Args:
            idx (int): Index of the data point.

        Returns:
            tuple: Tuple containing the image and its corresponding label.
        """
        annot_info = self.dataframe.iloc[idx]
        img_path = os.path.join(self.data_dir, str(annot_info['file_name']))
        annot_id = annot_info['id']
        label = annot_info['category_id']

        # Check cache for precomputed mask and skeleton
        cache_exists = False
        mask_filename = os.path.join(self.mask_dir, f"{annot_id}.png")
        if self.skeleton:
            skeleton_filename = os.path.join(self.skeleton_dir, f"{annot_id}.npy")
        if os.path.exists(mask_filename):
            masked_image = Image.open(mask_filename)
            cache_exists = True
            if self.skeleton:
                if os.path.exists(skeleton_filename):
                    skeleton_channel = np.load(skeleton_filename)
                    cache_exists = True
                else:
                    cache_exists = False
        # No cache exists, compute mask and skeleton
        if not cache_exists:
            image = Image.open(img_path).convert("RGB")

            if self.skeleton:
                keypoints = annot_info['keypoints']
                # Convert keypoints from string to list if necessary
                if isinstance(keypoints, str):
                    keypoints = ast.literal_eval(keypoints)
                connections = self.skeleton_category.get_connections()
                # Convert connections from string to list if necessary
                if isinstance(connections, str):
                    connections = ast.literal_eval(connections)
                skeleton_channel = create_skeleton_channel(keypoints, connections, height=image.size[0], width=image.size[1])

            # Extract bounding box and crop the image
            bbox = ast.literal_eval(annot_info['bbox'])
            x_min = math.floor(bbox[0])
            y_min = math.floor(bbox[1])
            w = math.ceil(bbox[2])
            h = math.ceil(bbox[3])
            bbox = [x_min, y_min, w, h]

            segmentation = ast.literal_eval(annot_info['segmentation'])
            mask = create_mask(image.size, segmentation)

            # masked_image = np.array(cropped_image) * np.expand_dims(cropped_mask, axis=2)
            masked_image = np.array(image) * np.expand_dims(mask, axis=2)
            masked_image = Image.fromarray(masked_image.astype('uint8'))

            # Crop the image and the mask to the bounding box
            masked_image = masked_image.crop((x_min, y_min, x_min + w, y_min + h))

            self.mask_cache[idx] = masked_image
            # Save mask
            masked_image.save(mask_filename)  # Save the cropped image as it is



            if self.skeleton:
                skeleton_channel = skeleton_channel[y_min:y_min + h, x_min:x_min + w]

                self.skeleton_cache[idx] = skeleton_channel
                # Save skeleton channel as a numpy file
                np.save(skeleton_filename, skeleton_channel)

        # resize, pad, transform (cached or newly computed images)
        if self.skeleton:
            # skeleton_channel = skeleton_channel[y_min:y_min + h, x_min:x_min + w]
            # print(skeleton_channel.shape)
            # print(skeleton_channel)
            masked_image, skeleton_channel = resize_and_pad(masked_image, self.size, skeleton_channel=skeleton_channel)
            masked_image = self.transform(masked_image, skeleton_channel)
        elif self.transform:
            masked_image = resize_and_pad(masked_image, self.size)
            masked_image = self.transform(masked_image)
        
        # return masked_image, label
        return masked_image, torch.tensor(label, dtype=torch.long)

    