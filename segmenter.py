import ultralytics
from ultralytics import YOLO, utils

import cv2
import os
from PIL import Image
import numpy as np

import matplotlib as plt
import pandas as pd
from ultralytics import YOLO

from shapely.geometry import Polygon
import ast


def create_mask(image_size, segmentation):
        """
        Creates a *binary mask* based on the segmentation of the object.

        Args:
            image_size (tuple): Size of the original image.
            segmentation (list): COCO-style segmentation of the object (flattened x y values).

        Returns:
            np.array: Binary mask of the object.
        """
        mask = np.zeros(image_size[::-1], dtype=np.uint8)
        for seg in segmentation:
            poly = np.array(seg).reshape((len(seg) // 2, 2)).astype(np.int32)
            cv2.fillPoly(mask, [poly], 1)
        return mask


def get_segs(polygon, image):
    segmentation = [polygon]

    poly = Polygon(polygon)
    area = poly.area
    min_x, min_y, max_x, max_y = poly.bounds
    bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

    seg = polygon.flatten().tolist()
    # [x, y, w, h] = cv2.boundingRect(polygon)
    return seg, bbox, area

def display_annotations(image,polygon=None):
    # Create a subplot
    fig, ax = plt.pyplot.subplots(1, 2, figsize=(10, 5))

    # Display the original image
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    if polygon is not None:
        # # Display the masked image
        # ax[1].imshow(mask_img)
        # ax[1].set_title('Segmentation Mask')
        # ax[1].axis('off')

        # Convert polygon points to integer
        polygon_points = np.array(polygon).reshape((-1, 1, 2)).astype(np.int32)

        image_np_copy = np.array(image).copy()
        cv2.polylines(image_np_copy, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

        ax[1].imshow(image_np_copy)
        ax[1].set_title('Polygon')
        ax[1].axis('off')

        # Show the plot
        plt.pyplot.show()

def add_segmentations(df, image_dir="", testing=False, cache_path=None, only_cache=False):
    """
    Uses the YOLOv8 model to make predictions for bbox and segmentations. 
    Bbox format is [x_min, y_min, width, height]
    Segmentations format is a flattened list of x , y values
    """

    # Load or initialize the cache DataFrame
    if cache_path and os.path.exists(cache_path):
        cache_df = pd.read_csv(cache_path)
    else:
        cache_df = pd.DataFrame(columns=df.columns)


    updated_rows = []
    for _, row in df.iterrows():
        # Check if the row exists in the cache and has 'segmentation' data
        if cache_path and 'path' in cache_df.columns:
            cached_row = cache_df[cache_df['path'] == row['path']]
            if not cached_row.empty and cached_row.iloc[0]['segmentation'] is not None and cached_row.iloc[0]['segmentation']:
                # Use cached data
                updated_rows.append(cached_row.iloc[0].to_dict())
                continue
        if cache_path and only_cache:
            continue

        # else:
        image_path = os.path.join(image_dir, row['path'])

        image = Image.open(image_path)
        og_width, og_height = image.size

        if 'bbox' in row and (isinstance(row['bbox'], (list, tuple, np.ndarray)) and pd.notnull(row['bbox']).all() or pd.notnull(row['bbox'])):
            bbox_exists = True
            print("running segmentation on pre-existing bbox")
            bbox = row['bbox']
            if isinstance(bbox, str):
                bbox_exists = True
                bbox = ast.literal_eval(bbox)  # Convert string to list
            if bbox == [0,0,0,0]:
                x, y, w, h = 0, 0, og_width, og_height
            else:
                x, y, w, h = bbox
            image = image.crop((x, y, x + w, y + h))
        elif 'bbox' in row and pd.notnull(row['bbox']):
            bbox = row['bbox']
            if isinstance(bbox, str):
                bbox_exists = True
                bbox = ast.literal_eval(bbox)  # Convert string to list
                if bbox == [0,0,0,0]:
                    x, y, w, h = 0, 0, og_width, og_height
                else:
                    x, y, w, h = bbox
                image = image.crop((x, y, x + w, y + h))
        else:
            bbox_exists = False

        # If no cache or segmentation is missing, run YOLOv8 model
        import torch
        from ultralytics.nn.tasks import SegmentationModel
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.modules.block import C2f, SPPF, C3, C3k2
        from torch.nn.modules.container import Sequential
        from torch.nn.modules.conv import Conv2d
        from torch.nn.modules.batchnorm import BatchNorm2d
        from torch.nn.modules.activation import SiLU
        from torch.nn.modules.pooling import MaxPool2d
        from torch.nn.modules.linear import Linear
        from torch.nn.modules.dropout import Dropout
        from ultralytics import YOLO

        # In the add_segmentations function, update the model loading part
        with torch.serialization.safe_globals([
            SegmentationModel,
            Sequential,
            Conv,
            Conv2d,
            BatchNorm2d,
            SiLU,
            MaxPool2d,
            Linear,
            Dropout,
            C2f,
            SPPF,
            C3,
            C3k2
        ]):
            model = YOLO('../checkpoints/yolo11x-seg.pt')
        # model = YOLO('../checkpoints/yolo11x-seg.pt')
        results = model(image)
        if len(results)>1:
            print("WARNING: Multiple objects detected in image")
        for result in results:
            if result.masks is None:
                print(f"No mask found for image: {row['path']}")
                # display_annotations(image)
                continue
            for mask in result.masks:
                polygon = mask.xy[0]

                # FOR VISUALISATION - uncomment
                # display_annotations(image, polygon)
                # Adjust segmentation coordinates to the original image's coordinate system
                if bbox_exists:
                    polygon = [(pt[0] + x, pt[1] + y) for pt in polygon]  # Shift by (x, y)
                    polygon = np.array(polygon) # Convert polygon back to a NumPy array

                segmentation, bbox, area = get_segs(polygon, image)
                iscrowd = 0 if len(results) <= 1 else 1

                updated_row = row.to_dict()
                updated_row['segmentation'] = [segmentation]
                updated_row['height'] = og_height
                updated_row['width'] = og_width
                updated_row['bbox'] = bbox
                updated_row['area'] = int(area)
                updated_row['iscrowd'] = iscrowd
                
                # Update the cache
                if cache_path:
                    cache_df = cache_df[cache_df['path'] != row['path']]  # Remove old row if exists
                    cache_df = pd.concat(
                        [cache_df, pd.DataFrame([updated_row])], ignore_index=True
                    )

                updated_rows.append(updated_row)

                break # only process the first result
            
    if cache_path and not only_cache:
        cache_df.to_csv(cache_path, index=False)

    updated_df = pd.DataFrame(updated_rows)
    return updated_df


def preprocess_lvl2(df, image_dir, yolo_model, testing=False):
    model = YOLO(yolo_model)
    # Process images and update the DataFrame with segmentation and size information
    df = add_segmentations(df, image_dir, model)
    print("Updated DataFrame with segmentations.")

    return df