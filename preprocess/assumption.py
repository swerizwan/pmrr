import os
import cv2
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from utils.smooth_bbox_utils import get_all_bbox_params
from utils.image_utils import get_single_image_crop_demo

class Inference(Dataset):
    def __init__(self, image_folder, frames, bboxes=None, joints2d=None, scale=1.0, crop_size=224, pre_load_imgs=None):
        """
        Initialize the Inference dataset.

        Args:
        - image_folder (str): Path to the folder containing image files.
        - frames (list): List of frame indices to consider.
        - bboxes (np.array, optional): Array of bounding boxes for each frame.
        - joints2d (np.array, optional): Array of 2D joints for each frame.
        - scale (float, optional): Scaling factor for the bounding boxes.
        - crop_size (int, optional): Size of the cropped image.
        - pre_load_imgs (np.array, optional): Pre-loaded images array (if any).
        """
        self.pre_load_imgs = pre_load_imgs
        
        if pre_load_imgs is None:
            # Load image file names from the image folder
            self.image_file_names = [
                osp.join(image_folder, x)
                for x in os.listdir(image_folder)
                if x.endswith('.png') or x.endswith('.jpg')
            ]
            # Sort image file names and select frames based on indices
            self.image_file_names = sorted(self.image_file_names)
            self.image_file_names = np.array(self.image_file_names)[frames]
        
        # Assign other attributes
        self.bboxes = bboxes
        self.joints2d = joints2d
        self.scale = scale
        self.crop_size = crop_size
        self.frames = frames
        self.has_keypoints = True if joints2d is not None else False
        
        self.norm_joints2d = np.zeros_like(self.joints2d)
        
        if self.has_keypoints:
            # Calculate bounding box parameters and normalize
            bboxes, time_pt1, time_pt2 = get_all_bbox_params(joints2d, vis_thresh=0.3)
            bboxes[:, 2:] = 150. / bboxes[:, 2:]
            self.bboxes = np.stack([bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 2]]).T
            
            # Update attributes based on time points
            self.image_file_names = self.image_file_names[time_pt1:time_pt2]
            self.joints2d = joints2d[time_pt1:time_pt2]
            self.frames = frames[time_pt1:time_pt2]

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
        - int: Number of samples (length of bboxes).
        """
        return len(self.bboxes)

    def __getitem__(self, idx):
        """
        Fetches a single sample from the dataset.

        Args:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - tuple: If has_keypoints=True, returns (normalized_image, keypoints_2d).
                 Otherwise, returns normalized_image only.
        """
        if self.pre_load_imgs is not None:
            img = self.pre_load_imgs
        else:
            # Load image from file based on index
            img = cv2.cvtColor(cv2.imread(self.image_file_names[idx]), cv2.COLOR_BGR2RGB)

        # Fetch bounding box and 2D joints (if available) for the current index
        bbox = self.bboxes[idx]
        j2d = self.joints2d[idx] if self.has_keypoints else None

        # Perform image cropping and normalization
        norm_img, raw_img, kp_2d = get_single_image_crop_demo(
            img,
            bbox,
            kp_2d=j2d,
            scale=self.scale,
            crop_size=self.crop_size)

        if self.has_keypoints:
            # Return normalized image and 2D keypoints
            return norm_img, kp_2d
        else:
            # Return normalized image only
            return norm_img



class ImageFolder(Dataset):
    def __init__(self, image_folder):
        """
        Custom dataset class for loading images from a folder.

        Args:
        - image_folder (str): Path to the folder containing images.
        """
        # Get list of image file names ending with .png or .jpg
        self.image_file_names = [
            osp.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ]
        # Sort the list of file names
        self.image_file_names = sorted(self.image_file_names)

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.image_file_names)

    def __getitem__(self, idx):
        """
        Fetches and preprocesses the image at the given index.

        Args:
        - idx (int): Index of the image to fetch.

        Returns:
        - tensor: Preprocessed image as a PyTorch tensor.
        """
        # Read image using OpenCV and convert BGR to RGB
        img = cv2.cvtColor(cv2.imread(self.image_file_names[idx]), cv2.COLOR_BGR2RGB)
        # Convert the image to PyTorch tensor
        to_tensor = ToTensor()
        img_tensor = to_tensor(img)
        return img_tensor
