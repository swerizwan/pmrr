def get_keypoints():
    """
    Get the COCO keypoints and their left/right flip correspondence map.

    Returns:
        keypoints (list): List of COCO keypoints in a specific order.
        keypoint_flip_map (dict): Dictionary mapping keypoints to their corresponding flipped keypoints.
                                  For keypoints not listed here, they are assumed to be symmetric and thus
                                  don't need explicit mapping.
    """
    # Keypoints are not available in the COCO json for the test split, so we
    # provide them here.
    keypoints = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    
    # Map for left/right keypoints that need explicit flipping definition.
    keypoint_flip_map = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }
    
    return keypoints, keypoint_flip_map
