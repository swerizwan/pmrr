import os
import cv2
import torch

import random
import numpy as np
import torchvision.transforms as transforms
from skimage.util.shape import view_as_windows


def get_image(filename):
    """
    Reads an image from a file and converts its color from RGB to BGR.
    
    Args:
        filename (str): The path to the image file.
        
    Returns:
        numpy.ndarray: The image in BGR color space.
    """
    # Read the image from the file
    image = cv2.imread(filename)
    # Convert the image color from RGB to BGR
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def do_augmentation(scale_factor=0.3, color_factor=0.2):
    """
    Generates random augmentation parameters for scaling, rotation, flipping, and color adjustment.
    
    Args:
        scale_factor (float): The maximum scaling factor.
        color_factor (float): The maximum color adjustment factor.
        
    Returns:
        tuple: A tuple containing the scale, rotation angle, flip flag, and color scale factors.
    """
    # Generate a random scaling factor within the specified range
    scale = random.uniform(1.2, 1.2 + scale_factor)
    
    # Rotation angle is set to 0 (disabled); it can be randomized within a range if needed
    rot = 0
    
    # Flip augmentation is disabled (set to False); it can be enabled and randomized if needed
    do_flip = False
    
    # Calculate upper and lower bounds for color adjustment
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    
    # Generate random color scale factors for each color channel
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]
    
    return scale, rot, do_flip, color_scale

def trans_point2d(pt_2d, trans):
    """
    Applies a 2D transformation to a point.
    
    Args:
        pt_2d (tuple): The original 2D point (x, y).
        trans (numpy.ndarray): The 3x3 transformation matrix.
        
    Returns:
        numpy.ndarray: The transformed 2D point (x, y).
    """
    # Create a homogeneous coordinate for the point (x, y, 1)
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    
    # Apply the transformation matrix to the point
    dst_pt = np.dot(trans, src_pt)
    
    # Return the transformed point in 2D
    return dst_pt[0:2]

def rotate_2d(pt_2d, rot_rad):
    """
    Rotates a 2D point around the origin by a given angle in radians.
    
    Args:
        pt_2d (tuple): The original 2D point (x, y).
        rot_rad (float): The rotation angle in radians.
        
    Returns:
        numpy.ndarray: The rotated 2D point (x', y').
    """
    # Extract the x and y coordinates of the point
    x = pt_2d[0]
    y = pt_2d[1]
    
    # Calculate the sine and cosine of the rotation angle
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    
    # Compute the new coordinates after rotation
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    
    # Return the rotated point as a numpy array
    return np.array([xx, yy], dtype=np.float32)


def rotate_2d(pt_2d, rot_rad):
    """Rotate a 2D point by a given angle in radians."""
    x, y = pt_2d
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array([x * cs - y * sn, x * sn + y * cs], dtype=np.float32)

def trans_point2d(point, trans):
    """Transform a 2D point using an affine transformation matrix."""
    src_pt = np.array([point[0], point[1], 1.0]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]

def do_augmentation():
    """Generate random augmentation parameters."""
    scale = np.random.uniform(1.0, 1.5)
    rot = np.random.uniform(-30, 30)
    do_flip = np.random.choice([True, False])
    color_scale = np.random.uniform(0.8, 1.2, 3)
    return scale, rot, do_flip, color_scale

def convert_cvimg_to_tensor(cvimg):
    """Convert a CV image to a tensor."""
    img = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # Convert HWC to CHW
    tensor = torch.from_numpy(img)
    return tensor

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale

    # compute source center
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y

    # compute rotation in radians
    rot_rad = np.pi * rot / 180

    # compute source direction vectors after rotation
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    # compute destination width, height, and center
    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)

    # destination direction vectors
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    # define source and destination points
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    # compute affine transformation
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def generate_patch_image_cv(cvimg, c_x, c_y, bb_width, bb_height, patch_width, patch_height, do_flip, scale, rot):
    # copy the input image to avoid modifications
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    # flip image if required
    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1

    # generate transformation matrix
    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot, inv=False)

    # apply the affine transformation to get the image patch
    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return img_patch, trans

def crop_image(image, kp_2d, center_x, center_y, width, height, patch_width, patch_height, do_augment):
    # get augmentation parameters
    if do_augment:
        scale, rot, do_flip, color_scale = do_augmentation()
    else:
        scale, rot, do_flip, color_scale = 1.3, 0, False, [1.0, 1.0, 1.0]

    # generate image patch
    image, trans = generate_patch_image_cv(
        image,
        center_x,
        center_y,
        width,
        height,
        patch_width,
        patch_height,
        do_flip,
        scale,
        rot
    )

    # transform keypoints according to the transformation
    for n_jt in range(kp_2d.shape[0]):
        kp_2d[n_jt] = trans_point2d(kp_2d[n_jt], trans)

    return image, kp_2d, trans

def transfrom_keypoints(kp_2d, center_x, center_y, width, height, patch_width, patch_height, do_augment):
    # get augmentation parameters
    if do_augment:
        scale, rot, do_flip, color_scale = do_augmentation()
    else:
        scale, rot, do_flip, color_scale = 1.2, 0, False, [1.0, 1.0, 1.0]

    # generate transformation matrix
    trans = gen_trans_from_patch_cv(
        center_x,
        center_y,
        width,
        height,
        patch_width,
        patch_height,
        scale,
        rot,
        inv=False,
    )

    # transform keypoints according to the transformation
    for n_jt in range(kp_2d.shape[0]):
        kp_2d[n_jt] = trans_point2d(kp_2d[n_jt], trans)

    return kp_2d, trans

def get_image_crops(image_file, bboxes):
    # read and convert image
    image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
    crop_images = []

    # iterate over bounding boxes
    for bb in bboxes:
        c_y, c_x = (bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2
        h, w = bb[2] - bb[0], bb[3] - bb[1]
        w = h = np.where(w / h > 1, w, h)  # ensure square aspect ratio

        # generate image patch
        crop_image, _ = generate_patch_image_cv(
            cvimg=image.copy(),
            c_x=c_x,
            c_y=c_y,
            bb_width=w,
            bb_height=h,
            patch_width=224,
            patch_height=224,
            do_flip=False,
            scale=1.3,
            rot=0,
        )
        # convert image patch to tensor
        crop_image = convert_cvimg_to_tensor(crop_image)
        crop_images.append(crop_image)

    # create a batch of image tensors
    batch_image = torch.cat([x.unsqueeze(0) for x in crop_images])
    return batch_image


def get_single_image_crop(image, bbox, scale=1.3):
    """
    Crop a single image based on the bounding box and scale.

    Args:
        image (str, torch.Tensor, or np.ndarray): Input image. Can be a file path, a Torch tensor, or a NumPy array.
        bbox (list or tuple): Bounding box coordinates in the format [x_center, y_center, width, height].
        scale (float, optional): Scaling factor for the bounding box. Default is 1.3.

    Returns:
        torch.Tensor: Cropped image as a Torch tensor.
    """
    # Check if the image is a file path
    if isinstance(image, str):
        if os.path.isfile(image):
            # Read and convert the image from BGR to RGB format
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        else:
            # Raise an exception if the file path is invalid
            print(image)
            raise BaseException(image, 'is not a valid file!')
    elif isinstance(image, torch.Tensor):
        # Convert Torch tensor to NumPy array
        image = image.numpy()
    elif not isinstance(image, np.ndarray):
        # Raise an exception for unknown image types
        raise('Unknown type for object', type(image))

    # Generate the cropped image using the given bounding box and scale
    crop_image, _ = generate_patch_image_cv(
        cvimg=image.copy(),
        c_x=bbox[0],
        c_y=bbox[1],
        bb_width=bbox[2],
        bb_height=bbox[3],
        patch_width=224,
        patch_height=224,
        do_flip=False,
        scale=scale,
        rot=0,
    )

    # Convert the cropped image to a Torch tensor
    crop_image = convert_cvimg_to_tensor(crop_image)

    return crop_image

def get_single_image_crop_demo(image, bbox, kp_2d, scale=1.2, crop_size=224):
    """
    Crop a single image and transform keypoints based on the bounding box, scale, and crop size.

    Args:
        image (str, torch.Tensor, or np.ndarray): Input image. Can be a file path, a Torch tensor, or a NumPy array.
        bbox (list or tuple): Bounding box coordinates in the format [x_center, y_center, width, height].
        kp_2d (np.ndarray): 2D keypoints array with shape (num_keypoints, 2).
        scale (float, optional): Scaling factor for the bounding box. Default is 1.2.
        crop_size (int, optional): Size of the cropped image. Default is 224.

    Returns:
        tuple: (cropped image as a Torch tensor, raw cropped image as a NumPy array, transformed keypoints)
    """
    # Check if the image is a file path
    if isinstance(image, str):
        if os.path.isfile(image):
            # Read and convert the image from BGR to RGB format
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        else:
            # Raise an exception if the file path is invalid
            print(image)
            raise BaseException(image, 'is not a valid file!')
    elif isinstance(image, torch.Tensor):
        # Convert Torch tensor to NumPy array
        image = image.numpy()
    elif not isinstance(image, np.ndarray):
        # Raise an exception for unknown image types
        raise('Unknown type for object', type(image))

    # Generate the cropped image using the given bounding box, scale, and crop size
    crop_image, trans = generate_patch_image_cv(
        cvimg=image.copy(),
        c_x=bbox[0],
        c_y=bbox[1],
        bb_width=bbox[2],
        bb_height=bbox[3],
        patch_width=crop_size,
        patch_height=crop_size,
        do_flip=False,
        scale=scale,
        rot=0,
    )

    # Transform the 2D keypoints using the transformation matrix
    if kp_2d is not None:
        for n_jt in range(kp_2d.shape[0]):
            kp_2d[n_jt, :2] = trans_point2d(kp_2d[n_jt], trans)

    # Create a copy of the raw cropped image
    raw_image = crop_image.copy()

    # Convert the cropped image to a Torch tensor
    crop_image = convert_cvimg_to_tensor(crop_image)

    return crop_image, raw_image, kp_2d

def read_image(filename):
    """
    Read an image from a file, resize it to 224x224, and convert it to a Torch tensor.

    Args:
        filename (str): File path of the image.

    Returns:
        torch.Tensor: Resized image as a Torch tensor.
    """
    # Read and convert the image from BGR to RGB format
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    # Resize the image to 224x224
    image = cv2.resize(image, (224, 224))
    # Convert the image to a Torch tensor
    return convert_cvimg_to_tensor(image)

def convert_cvimg_to_tensor(image):
    """
    Convert a CV image (NumPy array) to a Torch tensor with default transformations.

    Args:
        image (np.ndarray): Input image as a NumPy array.

    Returns:
        torch.Tensor: Transformed image as a Torch tensor.
    """
    # Get the default image transformations
    transform = get_default_transform()
    # Apply the transformations to the image
    image = transform(image)
    return image

def torch_inv_normal(image):
    """
    Invert the normalization of a Torch tensor image using standard ImageNet mean and std.

    Args:
        image (torch.Tensor): Normalized image tensor with shape (batch_size, channels, height, width).

    Returns:
        torch.Tensor: Image tensor with inverted normalization.
    """
    # Invert the normalization using ImageNet mean and std
    image = image * torch.tensor([0.229, 0.224, 0.225], device=image.device).reshape(1, 3, 1, 1)
    image = image + torch.tensor([0.485, 0.456, 0.406], device=image.device).reshape(1, 3, 1, 1)
    # Clamp the values to be within [0, 1]
    image = image.clamp(0., 1.)
    return image

def torch2numpy(image):
    """
    Converts a PyTorch tensor image to a NumPy array with pixel values in the range [0, 255].

    Args:
        image (torch.Tensor): The input image tensor with shape (C, H, W).

    Returns:
        np.ndarray: The output image as a NumPy array with shape (H, W, C) and dtype uint8.
    """
    # Detach the image tensor from the computation graph and move it to the CPU
    image = image.detach().cpu()
    
    # Define the inverse normalization transformation
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )
    
    # Apply the inverse normalization to the image
    image = inv_normalize(image)
    
    # Clamp the image values to the range [0, 1]
    image = image.clamp(0., 1.)
    
    # Convert the image to a NumPy array and scale pixel values to [0, 255]
    image = image.numpy() * 255.
    
    # Transpose the image to shape (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    
    # Convert the image to uint8 type
    return image.astype(np.uint8)

def torch_vid2numpy(video):
    """
    Converts a PyTorch tensor video to a NumPy array with pixel values in the range [0, 255].

    Args:
        video (torch.Tensor): The input video tensor with shape (N, C, T, H, W).

    Returns:
        np.ndarray: The output video as a NumPy array with shape (N, T, C, H, W) and dtype uint8.
    """
    # Detach the video tensor from the computation graph and move it to the CPU, then convert to a NumPy array
    video = video.detach().cpu().numpy()
    
    # Define the mean and std deviation for denormalization
    mean = np.array([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255])
    std = np.array([1 / 0.229, 1 / 0.224, 1 / 0.255])
    
    # Reshape mean and std to match the video dimensions
    mean = mean[np.newaxis, np.newaxis, ..., np.newaxis, np.newaxis]
    std = std[np.newaxis, np.newaxis, ..., np.newaxis, np.newaxis]
    
    # Apply denormalization
    video = (video - mean) / std
    
    # Clip the video values to the range [0, 1] and scale to [0, 255]
    video = video.clip(0., 1.) * 255
    
    # Convert the video to uint8 type
    video = video.astype(np.uint8)
    
    return video

def get_bbox_from_kp2d(kp_2d):
    """
    Calculates bounding boxes from 2D keypoints.

    Args:
        kp_2d (np.ndarray): Array of 2D keypoints with shape (N, K, 2) or (K, 2).

    Returns:
        np.ndarray: Bounding boxes with shape (N, 4) or (4,) for each set of keypoints.
    """
    # Determine if keypoints are batched or single set
    if len(kp_2d.shape) > 2:
        # Calculate upper left and lower right corners for batched keypoints
        ul = np.array([kp_2d[:, :, 0].min(axis=1), kp_2d[:, :, 1].min(axis=1)])  # upper left
        lr = np.array([kp_2d[:, :, 0].max(axis=1), kp_2d[:, :, 1].max(axis=1)])  # lower right
    else:
        # Calculate upper left and lower right corners for a single set of keypoints
        ul = np.array([kp_2d[:, 0].min(), kp_2d[:, 1].min()])  # upper left
        lr = np.array([kp_2d[:, 0].max(), kp_2d[:, 1].max()])  # lower right

    # Calculate the width and height of the bounding box
    w = lr[0] - ul[0]
    h = lr[1] - ul[1]
    
    # Calculate the center of the bounding box
    c_x, c_y = ul[0] + w / 2, ul[1] + h / 2
    
    # Adjust width and height to maintain aspect ratio and add margin
    w = h = np.where(w / h > 1, w, h)
    w = h = h * 1.1

    # Create the bounding box array
    bbox = np.array([c_x, c_y, w, h])  # shape = (4,N)
    
    return bbox

def normalize_2d_kp(kp_2d, crop_size=224, inv=False):
    """
    Normalizes 2D keypoints to a range of [-1, 1] or inversely scales them back.

    Args:
        kp_2d (np.ndarray): 2D keypoints with shape (K, 2) or (N, K, 2).
        crop_size (int): Size of the crop.
        inv (bool): If True, perform inverse normalization.

    Returns:
        np.ndarray: Normalized or inversely normalized keypoints.
    """
    if not inv:
        # Normalize keypoints to range [-1, 1]
        ratio = 1.0 / crop_size
        kp_2d = 2.0 * kp_2d * ratio - 1.0
    else:
        # Inversely normalize keypoints from range [-1, 1] back to original scale
        ratio = 1.0 / crop_size
        kp_2d = (kp_2d + 1.0) / (2 * ratio)

    return kp_2d

def get_default_transform():
    """
    Returns a default transformation pipeline for images.

    Returns:
        transforms.Compose: A composed transformation pipeline with normalization.
    """
    # Define the normalization transformation
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    # Create the transformation pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    return transform

def split_into_chunks(vid_names, seqlen, stride):
    """
    Splits video names into chunks based on sequence length and stride.

    Args:
        vid_names (np.ndarray): Array of video names.
        seqlen (int): Length of each sequence chunk.
        stride (int): Stride for moving window.

    Returns:
        list: List of start and end indices for each chunk.
    """
    video_start_end_indices = []

    # Find unique video names and their first occurrence indices
    video_names, group = np.unique(vid_names, return_index=True)
    
    # Sort video names based on their occurrence indices
    perm = np.argsort(group)
    video_names, group = video_names[perm], group[perm]

    # Split the indices of video frames based on unique video groups
    indices = np.split(np.arange(0, vid_names.shape[0]), group[1:])

    for idx in range(len(video_names)):
        indexes = indices[idx]
        
        # Skip if the number of frames is less than the sequence length
        if indexes.shape[0] < seqlen:
            continue
        
        # Create chunks of indices with the specified sequence length and stride
        chunks = view_as_windows(indexes, (seqlen,), step=stride)
        
        # Store the start and end indices of each chunk
        start_finish = chunks[:, (0, -1)].tolist()
        video_start_end_indices += start_finish

    return video_start_end_indices
