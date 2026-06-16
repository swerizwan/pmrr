from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json_tricks as json
import numpy as np

from .joints_datasets import JointsDataset


logger = logging.getLogger(__name__)


class COCODataset(JointsDataset):
    '''
    COCODataset class for loading COCO keypoints dataset.

    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
    "skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    '''

    def __init__(self, options, dataset, subset, use_augmentation=True, is_train=True):
        '''
        Initialize COCO dataset with specified options.

        Args:
            options (dict): Options for dataset configuration.
            dataset (str): Name of the dataset.
            subset (str): Subset of the dataset (train, val, test).
            use_augmentation (bool): Whether to use data augmentation (default is True).
            is_train (bool): Whether the dataset is used for training (default is True).
        '''
        super().__init__(options, dataset, subset, use_augmentation, is_train)
        
        # Initialize parameters
        self.in_vis_thre = 0.2  # Threshold for visibility of joints
        self.bbox_file = ''    # File for bounding box information (if applicable)
        self.use_gt_bbox = True  # Whether to use ground truth bounding boxes
        self.aspect_ratio = 1.0  # Aspect ratio of images
        self.pixel_std = 200    # Standard deviation of pixels
        
        # Load COCO annotations
        self.coco = COCO(self._get_ann_file_keypoint())

        # Deal with class names
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats  # Background + actual classes
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))  # Mapping from class name to index
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))  # Mapping from COCO category ID to class index
        self._coco_ind_to_class_ind = dict(
            [(self._class_to_coco_ind[cls], self._class_to_ind[cls]) for cls in self.classes[1:]]
        )  # Mapping from COCO category ID to internal class index

        # Load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        logger.info('=> num_images: {}'.format(self.num_images))

        # Constants for joints and skeletons
        self.num_joints = 17
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]  # Pairs of joints for flipping
        self.parent_ids = None  # Parent-child relationships for joints (if applicable)
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)  # IDs of joints in upper body
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)  # IDs of joints in lower body

        # Joint weights (used for loss calculation)
        self.joints_weight = np.array(
            [
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2,
                1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5
            ],
            dtype=np.float32
        ).reshape((self.num_joints, 1))

    def _get_ann_file_keypoint(self):
        '''
        Get annotation file path for COCO keypoints.

        Returns:
            str: File path to the annotation file.
        '''
        prefix = 'person_keypoints' if 'test' not in self.image_set else 'image_info'
        return os.path.join(self.root, 'annotations', prefix + '_' + self.image_set + '.json')

    def _load_image_set_index(self):
        '''
        Load image IDs from COCO dataset.

        Returns:
            list: List of image IDs.
        '''
        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_db(self):
        '''
        Get the database of ground truth data.

        Returns:
            list: List of ground truth annotations (database entries).
        '''
        if self.is_train or self.use_gt_bbox:
            # Use ground truth annotations
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # Use bounding boxes from detection
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        '''
        Load COCO keypoint annotations.

        Returns:
            list: List of dictionaries containing ground truth annotations.
        '''
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        '''
        Load COCO keypoint annotations for a specific image.

        Args:
            index (int): Index of the image.

        Returns:
            list: List of dictionaries containing annotations for the image.
        '''
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # Sanitize bounding boxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                continue

            # Ignore objects without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            # Prepare joint annotations
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            # Calculate center and scale
            center, scale = self._box2cs(obj['clean_bbox'][:4])

            # Record the annotation
            rec.append({
                'image': self.image_path_from_index(index),
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
            })

        return rec


    def _box2cs(self, box):
        """
        Convert bounding box to center and scale.

        Args:
            box (list): List containing [x, y, width, height].

        Returns:
            tuple: A tuple containing (center, scale).
                center (np.array): Array of shape (2,) representing the center of the box.
                scale (np.array): Array of shape (2,) representing the scale of the box.
        """
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        """
        Convert from [x, y, width, height] to (center, scale).

        Args:
            x (float): x-coordinate of top-left corner of the box.
            y (float): y-coordinate of top-left corner of the box.
            w (float): Width of the box.
            h (float): Height of the box.

        Returns:
            tuple: A tuple containing (center, scale).
                center (np.array): Array of shape (2,) representing the center of the box.
                scale (np.array): Array of shape (2,) representing the scale of the box.
        """
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        # Adjust aspect ratio
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        # Calculate scale
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)

        # Adjust scale if center is not -1
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def image_path_from_index(self, index):
        """
        Construct image path from index.

        Args:
            index (int): Index of the image.

        Returns:
            str: Path to the image.
        """
        file_name = '%012d.jpg' % index
        if '2014' in self.image_set:
            file_name = 'COCO_%s_' % self.image_set + file_name

        prefix = 'test2017' if 'test' in self.image_set else self.image_set

        data_name = prefix + '.zip@' if self.data_format == 'zip' else prefix

        image_path = os.path.join(
            self.root, 'images', data_name, file_name)

        return image_path

    def _load_coco_person_detection_results(self):
        """
        Load COCO person detection results.

        Returns:
            list: List of dictionaries containing detection results.
        """
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        logger.info('=> Total boxes: {}'.format(len(all_boxes)))

        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            img_name = self.image_path_from_index(det_res['image_id'])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.image_thre:
                continue

            num_boxes += 1

            center, scale = self._box2cs(box)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones((self.num_joints, 3), dtype=np.float)
            kpt_db.append({
                'image': img_name,
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })

        logger.info('=> Total boxes after filter low score@{}: {}'.format(
            self.image_thre, num_boxes))
        return kpt_db


    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path, ckp_name,
                *args, **kwargs):
        # Create a folder to store results if it doesn't exist
        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            try:
                os.makedirs(res_folder)
            except Exception:
                logger.error('Fail to make {}'.format(res_folder))

        # Define the path for saving results in JSON format
        res_file = os.path.join(
            res_folder, 'keypoints_{}_results_{}.json'.format(
                self.image_set, ckp_name)
        )

        # Prepare keypoints data in a structured format for each image
        _kpts = []
        for idx, kpt in enumerate(preds):
            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': int(img_path[idx][-16:-4])  # Extract image ID
            })

        # Organize keypoints by image ID using defaultdict
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        # Perform rescoring and Object Keypoint Similarity (OKS) Non-Maximum Suppression (NMS)
        num_joints = self.num_joints
        in_vis_thre = self.in_vis_thre
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                # Calculate average keypoint score based on visibility threshold
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score += t_s
                        valid_num += 1
                if valid_num != 0:
                    kpt_score /= valid_num
                # Rescore based on average keypoint score and box score
                n_p['score'] = kpt_score * box_score

            # Apply OKS NMS to retain the most relevant keypoints per image
            keep = list(range(len(img_kpts)))
            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        # Write the processed keypoints results to a JSON file
        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file)

        # Perform evaluation if not in 'test' mode and return evaluation metrics
        if 'test' not in self.image_set:
            info_str = self._do_python_keypoint_eval(
                res_file, res_folder)
            name_value = OrderedDict(info_str)
            return name_value, name_value['AP']
        else:
            # Return default values for 'test' mode
            return {'Null': 0}, 0

    def _write_coco_keypoint_results(self, keypoints, res_file):
        # Prepare data structure for COCO keypoints results
        data_pack = [
            {
                'cat_id': self._class_to_coco_ind[cls],
                'cls_ind': cls_ind,
                'cls': cls,
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if not cls == '__background__'
        ]

        # Generate results for one category using a kernel function
        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> writing results json to %s' % res_file)
        
        # Write results to JSON file with proper formatting
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        
        # Handle JSON formatting exceptions if any
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        # Extract necessary information from data pack
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        # Process each image's keypoints and format them for COCO evaluation
        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            # Extract keypoints data and format it properly
            _key_points = np.array([img_kpts[k]['keypoints']
                                    for k in range(len(img_kpts))])
            key_points = np.zeros(
                (_key_points.shape[0], self.num_joints * 3), dtype=np.float
            )

            for ipt in range(self.num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # Keypoints score

            # Prepare results with relevant information
            result = [
                {
                    'image_id': img_kpts[k]['image'],
                    'category_id': cat_id,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                    'center': list(img_kpts[k]['center']),
                    'scale': list(img_kpts[k]['scale'])
                }
                for k in range(len(img_kpts))
            ]
            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file, res_folder):
        # Load results for COCO keypoints evaluation
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Define names for different evaluation metrics
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        # Retrieve and format evaluation results
        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        return info_str
