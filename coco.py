import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from main.configs import cfg, parse_args
from main import constants, path_config
from preprocess import COCODataset
from models import hmr, SMPL, emo_body_lang
from utils.geometry_utils import perspective_projection
from utils.transforms_utils import transform_preds
from utils.uv_vis_utils import vis_smpl_iuv

import logging
logger = logging.getLogger(__name__)

# Define command-line arguments
parser = argparse.ArgumentParser()

# Argument to specify the path to the network checkpoint file. Default is None.
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')

# Argument to choose the evaluation dataset. Default is 'coco'.
parser.add_argument('--dataset', default='coco', help='Choose evaluation dataset')

# Argument to set the batch size for testing. Default is 32. Type is integer.
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for testing')

# Argument to specify the name of the SMPL regressor. Can be either 'hmr' or 'emo_body_lang'. Default is 'emo_body_lang'.
parser.add_argument('--regressor', type=str, choices=['hmr', 'emo_body_lang'], default='emo_body_lang', help='Name of the SMPL regressor.')

# Argument to provide the path to the configuration file for PMRR. Default is 'configs/emo_pose.yaml'.
parser.add_argument('--cfg_file', type=str, default='configs/emo_pose.yaml', help='Config file path for PMRR.')

# Argument to set the frequency of printing intermediate results. Default is 50. Type is integer.
parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')

# Argument to specify whether to shuffle the data. Default is False. When set, action is 'store_true' which means it will be True if the flag is present.
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')

# Argument to set the number of processes to use for data loading. Default is 8. Type is integer.
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')

# Argument to specify the file where detections will be saved if set. Default is None.
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')

# Argument to specify the output directory. Default is './notebooks/output/'.
parser.add_argument('--output_dir', type=str, default='./notebooks/output/', help='Output directory.')

# Argument to enable result visualization. Default is False. When set, action is 'store_true' which means it will be True if the flag is present.
parser.add_argument('--vis_demo', default=False, action='store_true', help='Result visualization')

# Argument to set the image size ratio for visualization. Default is 1. Type is integer.
parser.add_argument('--ratio', default=1, type=int, help='Image size ratio for visualization')

# Argument to specify the image name used for visualization. Default is an empty string.
parser.add_argument('--vis_imname', type=str, default='', help='Image name used for visualization.')

# Argument to specify other parameters. Default is None. Type is string. Accepts multiple values as a list.
parser.add_argument('--misc', default=None, type=str, nargs="*", help='Other parameters')


def run_evaluation(model, dataset_name, dataset, result_file,
                   batch_size=32, img_res=224, 
                   num_workers=32, shuffle=False, log_freq=50, options=None):
    """Run evaluation on the datasets and metrics we report in the paper. """

    # Determine whether to use a GPU or CPU for computation
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Transfer the model to the specified device
    model.to(device)

    # Load the SMPL model for body pose estimation
    smpl_neutral = SMPL(path_config.SMPL_MODEL_DIR, create_transl=False).to(device)
    
    # Check if results should be saved to a file
    save_results = result_file is not None

    # Disable shuffling if saving results to ensure consistent ordering
    if save_results:
        shuffle = False

    # Create a data loader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # Initialize arrays to store SMPL parameters and predictions
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    # Initialize variables to store prediction results
    num_joints = 17
    num_samples = len(dataset)
    print('dataset length: {}'.format(num_samples))
    all_preds = np.zeros((num_samples, num_joints, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for _, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
            # If visualizing a specific image name, filter the batches accordingly
            if len(options.vis_imname) > 0:
                imgnames = [i_n.split('/')[-1] for i_n in batch['imgname']]
                name_hit = False
                for i_n in imgnames:
                    if options.vis_imname in i_n:
                        name_hit = True
                        print('vis: ' + i_n)
                if not name_hit:
                    continue

            # Transfer the images to the specified device
            images = batch['img'].to(device)

            # Retrieve scale and center information from the batch
            scale = batch['scale'].numpy()
            center = batch['center'].numpy()

            # Number of images in the current batch
            num_images = images.size(0)

            # Retrieve ground truth 2D keypoints
            gt_keypoints_2d = batch['keypoints']
            # De-normalize 2D keypoints from [-1,1] to pixel space
            gt_keypoints_2d_orig = gt_keypoints_2d.clone()
            gt_keypoints_2d_orig[:, :, :-1] = 0.5 * img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)

            # Perform forward pass using the appropriate regressor
            if options.regressor == 'hmr':
                pred_rotmat, pred_betas, pred_camera = model(images)
            elif options.regressor == 'emo_body_lang':
                preds_dict, _ = model(images)
                pred_rotmat = preds_dict['smpl_out'][-1]['rotmat'].contiguous().view(-1, 24, 3, 3)
                pred_betas = preds_dict['smpl_out'][-1]['theta'][:, 3:13].contiguous()
                pred_camera = preds_dict['smpl_out'][-1]['theta'][:, :3].contiguous()

            # Generate SMPL output using predicted parameters
            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                        global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)

            # Extract the predicted joints
            pred_J24 = pred_output.joints[:, -24:]
            pred_JCOCO = pred_J24[:, constants.J24_TO_JCOCO]

            # Convert Weak Perspective Camera parameters to full perspective camera translation
            pred_cam_t = torch.stack([pred_camera[:, 1],
                                      pred_camera[:, 2],
                                      2 * constants.FOCAL_LENGTH / (img_res * pred_camera[:, 0] + 1e-9)], dim=-1)
            camera_center = torch.zeros(len(pred_JCOCO), 2, device=pred_camera.device)

            # Perform perspective projection to obtain 2D keypoints
            pred_keypoints_2d = perspective_projection(pred_JCOCO,
                                                       rotation=torch.eye(3, device=pred_camera.device).unsqueeze(0).expand(len(pred_JCOCO), -1, -1),
                                                       translation=pred_cam_t,
                                                       focal_length=constants.FOCAL_LENGTH,
                                                       camera_center=camera_center)

            # Transform the predicted keypoints to pixel space
            coords = pred_keypoints_2d + (img_res / 2.)
            coords = coords.cpu().numpy()

            # Retrieve ground truth 2D keypoints in COCO format
            gt_keypoints_coco = gt_keypoints_2d_orig[:, -24:][:, constants.J24_TO_JCOCO]
            vert_errors_batch = []
            for i, (gt2d, pred2d) in enumerate(zip(gt_keypoints_coco.cpu().numpy(), coords.copy())):
                # Compute per-vertex error for visualization
                vert_error = np.sqrt(np.sum((gt2d[:, :2] - pred2d[:, :2]) ** 2, axis=1))
                vert_error *= gt2d[:, 2]
                vert_mean_error = np.sum(vert_error) / np.sum(gt2d[:, 2] > 0)
                vert_errors_batch.append(10 * vert_mean_error)

            # Visualize the results if required
            if options.vis_demo:
                imgnames = [i_n.split('/')[-1] for i_n in batch['imgname']]
                if options.regressor == 'hmr':
                    iuv_pred = None

                images_vis = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
                images_vis = images_vis + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
                vis_smpl_iuv(images_vis.cpu().numpy(), pred_camera.cpu().numpy(), pred_output.vertices.cpu().numpy(),
                             smpl_neutral.faces, iuv_pred,
                             vert_errors_batch, imgnames, os.path.join('./notebooks/output/demo_results', dataset_name,
                                                                        options.checkpoint.split('/')[-3]), options)

            # Store the predictions
            preds = coords.copy()

            scale_ = np.array([scale, scale]).transpose()

            # Transform predictions back to original image space
            for i in range(coords.shape[0]):
                preds[i] = transform_preds(
                    coords[i], center[i], scale_[i], [img_res, img_res]
                )

            # Store predictions and related information
            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = 1.
            all_boxes[idx:idx + num_images, 5] = 1.
            image_path.extend(batch['imgname'])

            idx += num_images

        # If visualizing a specific image, exit after processing
        if len(options.vis_imname) > 0:
            exit()

        # Determine the checkpoint name for evaluation
        if args.checkpoint is None or 'model_checkpoint.pt' in args.checkpoint:
            ckp_name = 'spin_model'
        else:
            ckp_name = args.checkpoint.split('/')
            ckp_name = ckp_name[2].split('_')[1] + '_' + ckp_name[-1].split('.')[0]

        # Evaluate the predictions using the dataset's evaluation method
        name_values, perf_indicator = dataset.evaluate(
            cfg, all_preds, options.output_dir, all_boxes, image_path, ckp_name,
            filenames, imgnums
        )

        # Print evaluation results
        model_name = options.regressor
        if isinstance(name_values, list):
            for name_value in name_values:
                print_name_value(name_value, model_name)
        else:
            print_name_value(name_values, model_name)

    # Save reconstructions to a file for further processing
    if save_results:
        np.savez(result_file, pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera)


# Function to print formatted output in markdown table format
def print_name_value(name_value, full_arch_name):

    """
    Print a dictionary of names and values in markdown table format.

    Args:
        name_value (dict): A dictionary where keys are column names and values are the corresponding values.
        full_arch_name (str): The name of the architecture to be displayed in the first column.

    """

    names = name_value.keys()
    values = name_value.values()

    # Print header
    print(f"Architecture: {full_arch_name}")

    # Print each name and value
    for name, value in zip(names, values):
        print(f"- {name}: {value:.3f}")

if __name__ == '__main__':
    # Parse command line arguments
    args = parser.parse_args()
    parse_args(args)

    # Load the appropriate model based on the regressor argument
    if args.regressor == 'emo_body_lang':
        model = emo_body_lang(path_config.SMPL_MEAN_PARAMS, pretrained=True)
    if args.regressor == 'hmr':
        model = hmr(path_config.SMPL_MEAN_PARAMS)

    # Load checkpoint if provided and update model state
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'], strict=True)

    model.eval()  # Set the model to evaluation mode

    # Initialize the dataset for evaluation
    dataset = COCODataset(None, args.dataset, 'val2014', is_train=False)

    # Run evaluation on the dataset
    args.result_file = None
    run_evaluation(model, args.dataset, dataset, args.result_file,
                   batch_size=args.batch_size,
                   shuffle=args.shuffle,
                   log_freq=args.log_freq, options=args)

    # Print the model, checkpoint, and dataset used for evaluation
    print('{}: {}, {}'.format(args.regressor, args.checkpoint, args.dataset))
