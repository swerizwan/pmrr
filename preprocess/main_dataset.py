from __future__ import division

import cv2
import torch
import random
import numpy as np
from os.path import join
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from main import path_config, constants
from main.configs import cfg
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, transform_pts, rot_aa
from models.smpl import SMPL

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/path_config.py.
    """

    def __init__(self, options, dataset, ignore_3d=False, use_augmentation=True, is_train=True):
        super().__init__()
        
        # Initialize parameters
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = path_config.DATASET_FOLDERS[dataset]  # Directory containing dataset images
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)  # Image normalization parameters

        # Load dataset file based on train/eval mode
        if not is_train and dataset == 'h36m-p2' and options.eval_pve:
            self.data = np.load(path_config.DATASET_FILES[is_train]['h36m-p2-mosh'], allow_pickle=True)
        else:
            self.data = np.load(path_config.DATASET_FILES[is_train][dataset], allow_pickle=True)

        self.imgname = self.data['imgname']  # Image file names
        self.dataset_dict = {dataset: 0}  # Dictionary to map datasets for potential use

        logger.info('len of {}: {}'.format(self.dataset, len(self.imgname)))  # Log dataset size

        # Attempt to load ground truth mask paths, if available
        try:
            self.maskname = self.data['maskname']
        except KeyError:
            pass
        
        # Attempt to load part segmentation paths, if available
        try:
            self.partname = self.data['partname']
        except KeyError:
            pass

        # Load bounding box parameters (center and scale format)
        self.scale = self.data['scale']
        self.center = self.data['center']
        
        # Augmentation flag
        self.use_augmentation = use_augmentation
        
        # Attempt to load ground truth SMPL parameters (pose and shape), if available
        try:
            self.pose = self.data['pose'].astype(np.float)  # SMPL pose parameters
            self.betas = self.data['shape'].astype(np.float)  # SMPL shape parameters

            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']  # Availability of SMPL data
            else:
                self.has_smpl = np.ones(len(self.imgname), dtype=np.float32)
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname), dtype=np.float32)
        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname), dtype=np.float32)

        # Attempt to load SMPL 2D keypoints, if available
        try:
            self.smpl_2dkps = self.data['smpl_2dkps']
            self.has_smpl_2dkps = 1
        except KeyError:
            self.has_smpl_2dkps = 0

        # Attempt to load ground truth 3D pose, if available
        try:
            self.pose_3d = self.data['S']
            self.has_pose_3d = 1
        except KeyError:
            self.has_pose_3d = 0
        if ignore_3d:
            self.has_pose_3d = 0
        
        # Attempt to load 2D keypoints from ground truth and OpenPose predictions
        try:
            keypoints_gt = self.data['part']
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))  # Default to zeros if not available
        try:
            keypoints_openpose = self.data['openpose']
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))  # Default to zeros if not available
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)

        # Attempt to load gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)  # Default to -1 if not available
        
        self.length = self.scale.shape[0]  # Number of samples in the dataset

        # Initialize SMPL model
        self.smpl = SMPL(path_config.SMPL_MODEL_DIR,
                         batch_size=cfg.TRAIN.BATCH_SIZE,
                         create_transl=False)
        
        self.faces = self.smpl.faces  # Faces of the SMPL model


    def augm_params(self):
        """Get augmentation parameters."""
        # Initialize parameters for augmentation
        flip = 0            # flipping indicator
        pn = np.ones(3)     # per channel pixel-noise
        rot = 0             # rotation angle
        sc = 1              # scaling factor

        if self.is_train:
            # If in training mode, apply augmentations with certain probabilities

            # Flip the image with a probability of 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
            
            # Apply pixel noise uniformly in the range [1-noise_factor, 1+noise_factor] for each channel
            pn = np.random.uniform(1 - self.options.noise_factor, 1 + self.options.noise_factor, 3)
            
            # Sample rotation angle from a normal distribution, clipped to [-2*rotFactor, 2*rotFactor]
            rot = min(2 * self.options.rot_factor,
                      max(-2 * self.options.rot_factor, np.random.randn() * self.options.rot_factor))
            
            # Sample scaling factor from a normal distribution, clipped to [1-scale_factor, 1+scale_factor]
            sc = min(1 + self.options.scale_factor,
                     max(1 - self.options.scale_factor, np.random.randn() * self.options.scale_factor + 1))

            # Set rotation to 0 with a probability of 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
        
        # Return the augmentation parameters
        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process RGB image and apply augmentations."""
        # Crop and resize the image based on the center, scale, and rotation
        rgb_img = crop(rgb_img, center, scale, [constants.IMG_RES, constants.IMG_RES], rot=rot)

        # Flip the image horizontally if flip is set
        if flip:
            rgb_img = flip_img(rgb_img)

        # Add per channel pixel noise
        rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
        rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
        rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))

        # Normalize the image to the range [0,1] and transpose to (3,224,224)
        rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f, is_smpl=False):
        """Process ground truth 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            # Transform keypoints based on center, scale, and rotation
            kp[i, 0:2] = transform(kp[i, 0:2] + 1, center, scale, [constants.IMG_RES, constants.IMG_RES], rot=r)
        
        # Convert keypoints to normalized coordinates in the range [-1, 1]
        kp[:, :-1] = 2. * kp[:, :-1] / constants.IMG_RES - 1.
        
        # Flip the x-coordinates if flip is set
        if f:
            kp = flip_kp(kp, is_smpl)
        
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f, is_smpl=False):
        """Process ground truth 3D keypoints and apply all augmentation transforms."""
        # Initialize the rotation matrix
        rot_mat = np.eye(3)
        if not r == 0:
            # Compute the rotation matrix for the given angle
            rot_rad = -r * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
        
        # Apply the rotation to the 3D keypoints
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1])

        # Flip the x-coordinates if flip is set
        if f:
            S = flip_kp(S, is_smpl)
        
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters and apply all augmentation transforms."""
        # Rotate the first three elements of the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        
        # Flip the pose parameters if flip is set
        if f:
            pose = flip_pose(pose)
        
        pose = pose.astype('float32')
        return pose

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()

        # Load image
        imgname = join(self.img_dir, self.imgname[index])
        try:
            img = cv2.imread(imgname)[:, :, ::-1].copy().astype(np.float32)
            orig_shape = np.array(img.shape)[:2]
        except:
            logger.error('Failed to load {}'.format(imgname))

        kp_is_smpl = True if self.dataset == 'surreal' else False

        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
            pose = self.pose_processing(pose, rot, flip)
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        # Process image
        img = self.rgb_processing(img, center, sc * scale, rot, flip, pn)
        img = torch.from_numpy(img).float()

        # Store image before normalization to use it in visualization
        item['img'] = self.normalize_img(img)
        item['pose'] = torch.from_numpy(pose).float()
        item['betas'] = torch.from_numpy(betas).float()
        item['imgname'] = imgname

        # Process SMPL 2D keypoints, if available
        if self.has_smpl_2dkps:
            smpl_2dkps = self.smpl_2dkps[index].copy()
            smpl_2dkps = self.j2d_processing(smpl_2dkps, center, sc * scale, rot, f=0)
            smpl_2dkps[smpl_2dkps[:, 2] == 0] = 0
            if flip:
                smpl_2dkps = smpl_2dkps[constants.SMPL_JOINTS_FLIP_PERM]
                smpl_2dkps[:, 0] = - smpl_2dkps[:, 0]
            item['smpl_2dkps'] = torch.from_numpy(smpl_2dkps).float()
        else:
            item['smpl_2dkps'] = torch.zeros(24, 3, dtype=torch.float32)

        # Get 3D pose, if available
        if self.has_pose_3d:
            S = self.pose_3d[index].copy()
            item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip, kp_is_smpl)).float()
        else:
            item['pose_3d'] = torch.zeros(24, 4, dtype=torch.float32)

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.keypoints[index].copy()
        item['keypoints'] = torch.from_numpy(self.j2d_processing(keypoints, center, sc * scale, rot, flip, kp_is_smpl)).float()

        # Store additional metadata
        item['has_smpl'] = self.has_smpl[index]
        item['has_pose_3d'] = self.has_pose_3d
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)
        item['orig_shape'] = orig_shape
        item['is_flipped'] = flip
        item['rot_angle'] = np.float32(rot)
        item['gender'] = self.gender[index]
        item['sample_index'] = index
        item['dataset_name'] = self.dataset

        try:
            item['maskname'] = self.maskname[index]
        except AttributeError:
            item['maskname'] = ''
        try:
            item['partname'] = self.partname[index]
        except AttributeError:
            item['partname'] = ''

        return item

    def __len__(self):
        return len(self.imgname)
