#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRED RGB-Event Fusion Dataset DataLoader

This module provides a PyTorch dataset loader for the FRED RGB-Event fusion dataset.
It loads both RGB and Event modalities along with their annotations.

Author: Mi Code
Version: 1.0
Date: 2025-11-09
"""

import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image


class FREDFusionDataset(Dataset):
    """FRED RGB-Event Fusion Dataset"""
    
    def __init__(self, annotation_path, fred_root, input_shape=(640, 640), train=True, augment=True):
        """
        Initialize FRED Fusion Dataset
        
        Args:
            annotation_path: Path to fusion annotation JSON file
            fred_root: Root directory of FRED dataset
            input_shape: Input image shape (height, width)
            train: Whether this is training data
            augment: Whether to apply data augmentation
        """
        self.annotation_path = annotation_path
        self.fred_root = fred_root
        self.input_shape = input_shape
        self.train = train
        self.augment = augment and train
        
        # Load annotations
        print(f"Loading fusion dataset from: {annotation_path}")
        with open(annotation_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.images = self.data['images']
        self.annotations = self.data['annotations']
        self.categories = self.data['categories']
        
        # Organize annotations by image ID
        self.annotations_by_image = {}
        for ann in self.annotations:
            image_id = ann['image_id']
            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []
            self.annotations_by_image[image_id].append(ann)
        
        print(f"Loaded {len(self.images)} images with {len(self.annotations)} annotations")
        
        # Category mapping
        self.category_mapping = {cat['id']: i for i, cat in enumerate(self.categories)}
        self.num_classes = len(self.categories)
    
    def __len__(self):
        """Get dataset length"""
        return len(self.images)
    
    def __getitem__(self, index):
        """Get item by index"""
        # Get image info
        image_info = self.images[index]
        image_id = image_info['id']
        
        # Load RGB and Event images
        rgb_path = os.path.join(self.fred_root, image_info['rgb_file_name'])
        event_path = os.path.join(self.fred_root, image_info['event_file_name'])
        
        # Load images
        rgb_image = cv2.imread(rgb_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        event_image = cv2.imread(event_path)
        event_image = cv2.cvtColor(event_image, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        annotations = self.annotations_by_image.get(image_id, [])
        
        # Preprocess images and annotations
        rgb_image, event_image, targets = self._preprocess(
            rgb_image, event_image, annotations, image_info
        )
        
        return {
            'rgb': rgb_image,
            'event': event_image,
            'targets': targets,
            'image_info': image_info
        }
    
    def _preprocess(self, rgb_image, event_image, annotations, image_info):
        """Preprocess images and annotations"""
        height, width = rgb_image.shape[:2]
        
        # Convert to float32 and normalize
        rgb_image = rgb_image.astype(np.float32) / 255.0
        event_image = event_image.astype(np.float32) / 255.0
        
        # Resize images
        rgb_image = cv2.resize(rgb_image, (self.input_shape[1], self.input_shape[0]))
        event_image = cv2.resize(event_image, (self.input_shape[1], self.input_shape[0]))
        
        # Convert to CHW format
        rgb_image = np.transpose(rgb_image, (2, 0, 1))  # HWC -> CHW
        event_image = np.transpose(event_image, (2, 0, 1))  # HWC -> CHW
        
        # Convert to torch tensors
        rgb_image = torch.from_numpy(rgb_image).float()
        event_image = torch.from_numpy(event_image).float()
        
        # Prepare targets
        targets = self._prepare_targets(annotations, image_info, width, height)
        
        return rgb_image, event_image, targets
    
    def _prepare_targets(self, annotations, image_info, orig_width, orig_height):
        """Prepare targets for YOLO detection"""
        if len(annotations) == 0:
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64),
                'image_id': torch.tensor(image_info['id'], dtype=torch.int64)
            }
        
        boxes = []
        labels = []
        
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, w, h] in original image coordinates
            
            # Convert to normalized coordinates [0, 1]
            x1 = bbox[0] / orig_width
            y1 = bbox[1] / orig_height
            x2 = (bbox[0] + bbox[2]) / orig_width
            y2 = (bbox[1] + bbox[3]) / orig_height
            
            # Clamp to [0, 1]
            x1 = max(0, min(1, x1))
            y1 = max(0, min(1, y1))
            x2 = max(0, min(1, x2))
            y2 = max(0, min(1, y2))
            
            # Convert to input shape coordinates
            x1 = x1 * self.input_shape[1]
            y1 = y1 * self.input_shape[0]
            x2 = x2 * self.input_shape[1]
            y2 = y2 * self.input_shape[0]
            
            boxes.append([x1, y1, x2, y2])
            labels.append(self.category_mapping[ann['category_id']])
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        return {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor(image_info['id'], dtype=torch.int64)
        }
    
    def _apply_augmentation(self, rgb_image, event_image, boxes):
        """Apply data augmentation (placeholder for future implementation)"""
        # TODO: Implement data augmentation
        # - Random horizontal flip
        # - Random crop
        # - Color jittering for RGB
        # - Noise injection for Event
        return rgb_image, event_image, boxes
    
    def collate_fn(self, batch):
        """Custom collate function for batching"""
        rgb_images = torch.stack([item['rgb'] for item in batch])
        event_images = torch.stack([item['event'] for item in batch])
        
        # Handle variable number of targets per image
        targets = []
        image_infos = []
        
        for item in batch:
            targets.append(item['targets'])
            image_infos.append(item['image_info'])
        
        return {
            'rgb': rgb_images,
            'event': event_images,
            'targets': targets,
            'image_infos': image_infos
        }


def create_fusion_dataloader(annotation_path, fred_root, batch_size=16, input_shape=(640, 640), 
                           train=True, shuffle=True, num_workers=4, augment=True):
    """
    Create a fusion dataset dataloader
    
    Args:
        annotation_path: Path to fusion annotation JSON file
        fred_root: Root directory of FRED dataset
        batch_size: Batch size
        input_shape: Input image shape (height, width)
        train: Whether this is training data
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        augment: Whether to apply data augmentation
    
    Returns:
        DataLoader instance
    """
    dataset = FREDFusionDataset(
        annotation_path=annotation_path,
        fred_root=fred_root,
        input_shape=input_shape,
        train=train,
        augment=augment
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )
    
    return dataloader


# Test function
def test_fusion_dataloader():
    """Test the fusion dataloader"""
    print("Testing FRED Fusion DataLoader...")
    
    # Create dataloader
    dataloader = create_fusion_dataloader(
        annotation_path="datasets/fred_fusion/annotations/instances_train.json",
        fred_root="/mnt/data/datasets/fred",
        batch_size=4,
        input_shape=(640, 640),
        train=True,
        shuffle=True,
        num_workers=0
    )
    
    # Test one batch
    for batch in dataloader:
        print(f"RGB images shape: {batch['rgb'].shape}")
        print(f"Event images shape: {batch['event'].shape}")
        print(f"Number of targets: {len(batch['targets'])}")
        
        for i, targets in enumerate(batch['targets']):
            print(f"Image {i}: {len(targets['boxes'])} objects")
        
        break
    
    print("DataLoader test completed successfully!")


if __name__ == "__main__":
    test_fusion_dataloader()