from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np

from main import path_config
from .main_dataset import BaseDataset

logger = logging.getLogger(__name__)


class JointsDataset(BaseDataset):
    def __init__(self, options, dataset, subset, use_augmentation, is_train=True):
        """
        Initializes the JointsDataset instance.

        Args:
            options: Configuration options for the dataset.
            dataset: Name or identifier of the dataset.
            subset: Specific subset of the dataset (e.g., train, validation).
            use_augmentation: Boolean indicating if data augmentation should be used.
            is_train: Boolean indicating if the dataset is for training. Defaults to True.
        """
        super().__init__(options, dataset, use_augmentation=use_augmentation, is_train=is_train)
        
        # Initialize dataset-specific attributes
        self.num_joints = 0  # Number of joints in the dataset (to be set later)
        self.pixel_std = 200  # Standard pixel value for normalization
        self.flip_pairs = []  # Pairs of joints to be flipped during augmentation
        self.parent_ids = []  # Parent joint indices for hierarchical models

        self.is_train = is_train  # Flag indicating if the dataset is for training
        self.root = path_config.DATASET_FOLDERS[dataset]  # Root directory of the dataset
        self.image_set = subset  # Subset of the dataset (e.g., train, val)

        self.data_format = 'jpg'  # Format of the image files

    def _get_db(self):
        """
        Placeholder method for retrieving the dataset.
        Must be implemented by subclasses.
        
        Raises:
            NotImplementedError: Always raised to indicate this method should be overridden.
        """
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """
        Placeholder method for evaluating predictions.
        Must be implemented by subclasses.

        Args:
            cfg: Configuration for evaluation.
            preds: Predictions made by the model.
            output_dir: Directory to save evaluation results.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: Always raised to indicate this method should be overridden.
        """
        raise NotImplementedError

    # def __len__(self,):
    #     """
    #     Returns the length of the dataset.
    #     
    #     Returns:
    #         int: Number of samples in the dataset.
    #     """
    #     return len(self.db)

    def select_data(self, db):
        """
        Selects data from the dataset based on certain criteria.

        Args:
            db: The dataset to be filtered (list of dictionaries).

        Returns:
            list: The filtered dataset meeting the selection criteria.
        """
        db_selected = []  # Initialize an empty list to store selected records

        # Iterate over each record in the database
        for rec in db:
            num_vis = 0  # Number of visible joints
            joints_x = 0.0  # Sum of x-coordinates of visible joints
            joints_y = 0.0  # Sum of y-coordinates of visible joints

            # Iterate over each joint and its visibility
            for joint, joint_vis in zip(rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue  # Skip if the joint is not visible
                num_vis += 1
                joints_x += joint[0]
                joints_y += joint[1]

            if num_vis == 0:
                continue  # Skip if no joints are visible

            # Compute the average coordinates of visible joints
            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            # Calculate the area of the bounding box
            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)

            # Calculate the center of the joints and the bounding box
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])

            # Compute the Euclidean distance between joint center and bounding box center
            diff_norm2 = np.linalg.norm((joints_center - bbox_center), 2)

            # Compute a selection metric based on distance and area
            ks = np.exp(-1.0 * (diff_norm2**2) / ((0.2)**2 * 2.0 * area))
            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16

            # Select the record if it meets the selection criterion
            if ks > metric:
                db_selected.append(rec)

        # Log the number of records in the original and selected datasets
        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))

        return db_selected

