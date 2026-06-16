from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

def transform_preds(coords, center, scale, output_size):
    """
    Transform coordinates from original image space to output space.

    Args:
    - coords (np.array): Original coordinates to transform, shape (N, 2).
    - center (np.array): Center point of the object to transform around, shape (2,).
    - scale (float or np.array): Scale factor(s) relative to the object, can be scalar or (2,) array.
    - output_size (tuple): Output size (width, height).

    Returns:
    - target_coords (np.array): Transformed coordinates in the output space, shape (N, 2).
    """
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    """
    Generate affine transformation matrix.

    Args:
    - center (np.array): Center point of the object, shape (2,).
    - scale (float or np.array): Scale factor(s) relative to the object, can be scalar or (2,) array.
    - rot (float): Rotation angle in degrees.
    - output_size (tuple): Output size (width, height).
    - shift (np.array): Shift parameters for translation, shape (2,).
    - inv (int): Flag indicating inverse transformation.

    Returns:
    - trans (np.array): Affine transformation matrix, shape (2, 3).
    """
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    """
    Apply affine transformation matrix to a point.

    Args:
    - pt (np.array): Point to transform, shape (2,).
    - t (np.array): Affine transformation matrix, shape (2, 3).

    Returns:
    - new_pt (np.array): Transformed point, shape (2,).
    """
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    """
    Calculate the third point to form a triangle.

    Args:
    - a (np.array): First point, shape (2,).
    - b (np.array): Second point, shape (2,).

    Returns:
    - third (np.array): Third point to complete the triangle, shape (2,).
    """
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """
    Calculate direction vector after rotation.

    Args:
    - src_point (np.array): Source point, shape (2,).
    - rot_rad (float): Rotation angle in radians.

    Returns:
    - src_result (np.array): Transformed direction vector, shape (2,).
    """
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result
