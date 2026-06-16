import os
#os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import cv2
import time
import json
import torch
import joblib
import argparse
import numpy as np
from tqdm import tqdm
from multi_person_tracker import MPT
#from mpty import MPT8
from torch.utils.data import DataLoader
import os.path as osp
from matplotlib.image import imsave
from skimage.transform import resize
from torchvision.transforms import Normalize

from main.configs import cfg, parse_args
from models import hmr, emo_body_lang, SMPL
from main import path_config, constants
from preprocess.assumption import Inference
from utils.renderer import PyRenderer
from utils.imutils import crop
from utils.pose_tracker import run_posetracker
from utils.demo_utils import (
    download_url,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
)

MIN_NUM_FRAMES = 1

def process_image(img_file, input_res=224):
    """
    Read image, do preprocessing and possibly crop it according to the bounding box.
    
    Args:
    - img_file (str): File path to the image.
    - input_res (int): Desired input resolution for the image (default is 224).

    Returns:
    - img_np (numpy.ndarray): Cropped and resized image as a numpy array.
    - img (torch.Tensor): Preprocessed image as a PyTorch tensor.
    - norm_img (torch.Tensor): Normalized and preprocessed image ready for model input.
    """
    # Define normalization transformation
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    
    # Read image using OpenCV (BGR format), and convert to RGB
    img = cv2.imread(img_file)[:,:,::-1].copy()  # PyTorch does not support negative stride at the moment
    
    # Assume that the person is centered in the image
    height = img.shape[0]
    width = img.shape[1]
    center = np.array([width // 2, height // 2])
    
    # Scale factor based on the larger dimension
    scale = max(height, width) / 200
    
    # Crop and resize the image
    img_np = crop(img, center, scale, (input_res, input_res))  # Function crop() performs the cropping
    img_np = img_np.astype(np.float32) / 255.  # Convert to float32 and normalize to [0, 1]
    
    # Convert numpy array to PyTorch tensor and adjust the dimensions
    img = torch.from_numpy(img_np).permute(2, 0, 1)  # Permute dimensions to (channels, height, width)
    
    # Apply normalization
    norm_img = normalize_img(img.clone())[None]  # Clone tensor and apply normalization
    
    return img_np, img, norm_img


def run_image_demo(args):
    # Determine device (CPU or GPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define model
    if args.regressor == 'hmr-spin':
        # Initialize HMR-SPIN model
        model = hmr(path_config.SMPL_MEAN_PARAMS).to(device)
    elif args.regressor == 'emo_body_lang':
        # Initialize PMRR model
        model = emo_body_lang(path_config.SMPL_MEAN_PARAMS, pretrained=True).to(device)

    # Load pretrained weights
    if args.checkpoint is not None:
        # Load checkpoint weights if provided
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'], strict=True)

    # Load SMPL model for parameterized human body
    smpl = SMPL(path_config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    model.eval()

    # Setup renderer for visualization
    renderer = PyRenderer(resolution=(constants.IMG_RES, constants.IMG_RES))

    # Preprocess input image and generate predictions
    img_np, img, norm_img = process_image(args.img_file, input_res=constants.IMG_RES)

    with torch.no_grad():
        if args.regressor == 'hmr-spin':
            # Run HMR-SPIN model to predict body parameters
            pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
            # Predict 3D body model vertices using SMPL parameters
            pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices
        elif args.regressor == 'emo_body_lang':
            # Run PMRR model to predict body parameters
            preds_dict, _ = model(norm_img.to(device))
            output = preds_dict['smpl_out'][-1]
            pred_camera = output['theta'][:, :3]
            pred_vertices = output['verts']

    # Calculate camera parameters for rendering
    camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] + 1e-9)], dim=-1)
    camera_translation = camera_translation[0].cpu().numpy()
    pred_vertices = pred_vertices[0].cpu().numpy()

    # Convert input image to RGB (if not already)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # Render front-view shape
    save_mesh_path = None  # Optionally save mesh
    img_shape = renderer(
                    pred_vertices,
                    img=img_np,
                    cam=pred_camera[0].cpu().numpy(),
                    color_type='purple',
                    mesh_filename=save_mesh_path
                )

    # Render side views
    aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
    center = pred_vertices.mean(axis=0)
    rot_vertices = np.dot((pred_vertices - center), aroundy) + center
    
    # Render side-view shape
    img_shape_side = renderer(
                        rot_vertices,
                        img=np.ones_like(img_np),  # Background color for side view
                        cam=pred_camera[0].cpu().numpy(),
                        color_type='purple',
                        mesh_filename=save_mesh_path
                    )

    # Save rendered image
    output_path = os.path.join(args.output_folder, args.img_file.split('/')[-2])
    os.makedirs(output_path, exist_ok=True)

    img_name = os.path.basename(args.img_file).split('.')[0]
    save_name = os.path.join(output_path, img_name)
    
    # Save front-view and side-view images
    cv2.imwrite(save_name + '_pmrr.png', img_shape)
    cv2.imwrite(save_name + '_pmrr_side.png', img_shape_side)
    
    print(f'Saved the result image to {output_path}.')

def run_video_demo(args):
    # Determine device (CPU or GPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.image_folder is None:
        video_file = args.vid_file

        # Optional download the youtube video 
        if video_file.startswith('https://www.youtube.com'):
            print(f'Downloading YouTube video \"{video_file}\"')
            video_file = download_url(video_file, '/tmp')

            if video_file is None:
                exit('Youtube url is not valid!')

            print(f'YouTube Video has been downloaded to {video_file}...')

        # Check if the video file exists
        if not os.path.isfile(video_file):
            exit(f'Input video \"{video_file}\" does not exist!')
        
        # Determine output path for saving results
        output_path = os.path.join(args.output_folder, os.path.basename(video_file).replace('.mp4', ''))

        # Extract frames from the video and get information
        image_folder, num_frames, img_shape = video_to_images(video_file, return_info=True)
    else:
        # If image_folder is provided instead of a video file
        image_folder = args.image_folder
        num_frames = len(os.listdir(image_folder))
        img_shape = cv2.imread(os.path.join(image_folder, os.listdir(image_folder)[0])).shape

        # Determine output path for saving results
        output_path = os.path.join(args.output_folder, os.path.split(image_folder)[-1])

    # Create the output directory if it does not exist
    os.makedirs(output_path, exist_ok=True)

    # Print number of frames in the input video
    print(f'Input video number of frames: {num_frames}')

    # If not image_based, get original image dimensions
    if not args.image_based:
        orig_height, orig_width = img_shape[:2]

    # Measure total execution time
    total_time = time.time()

    # Run tracking
    bbox_scale = 1.0  # Initialize bounding box scale factor

    if args.use_gt:  # If using ground truth annotations
        with open(args.anno_file) as f:
            tracking_anno = json.load(f)  # Load tracking annotations from JSON
        tracking_results = {}  # Initialize tracking results dictionary
        for tracklet in tracking_anno:  # Iterate through each tracklet in annotations
            track_id = tracklet['idx']  # Extract track ID
            frames = tracklet['frames']  # Extract frames information for the tracklet
            f_id = []  # Initialize list for frame IDs
            bbox = []  # Initialize list for bounding boxes
            for f in frames:  # Iterate through each frame in the tracklet
                f_id.append(f['frame_id'])  # Append frame ID to list
                # Extract and scale bounding box coordinates
                x_tl, y_tl = f['rect']['tl']['x'] * orig_width, f['rect']['tl']['y'] * orig_height
                x_br, y_br = f['rect']['br']['x'] * orig_width, f['rect']['br']['y'] * orig_height

                x_c, y_c = (x_br + x_tl) / 2., (y_br + y_tl) / 2.  # Calculate center coordinates
                w, h = x_br - x_tl, y_br - y_tl  # Calculate width and height
                wh_max = max(w, h)  # Determine maximum dimension
                x_tl, y_tl = x_c - wh_max / 2., y_c - wh_max / 2.  # Adjust top-left corner coordinates

                bbox.append(np.array([x_c, y_c, wh_max, wh_max]))  # Append bounding box to list
            f_id = np.array(f_id)  # Convert frame IDs to NumPy array
            bbox = np.array(bbox)  # Convert bounding boxes to NumPy array
            tracking_results[track_id] = {'frames': f_id, 'bbox': bbox}  # Store results in dictionary

    else:  # If not using ground truth annotations
        # bbox_scale = 1.1  # Set bounding box scale factor (optional)
        if args.tracking_method == 'pose':  # If tracking method is pose-based
            if not os.path.isabs(video_file):  # If video file path is not absolute, make it absolute
                video_file = os.path.join(os.getcwd(), video_file)
            # Run pose tracker with specified parameters
            tracking_results = run_posetracker(video_file, staf_folder=args.staf_dir, display=args.display)
        else:  # If using another tracking method (e.g., multi-object tracker)
            # Initialize multi-object tracker (MOT) with specified parameters
            mot = MPT(
                device=device,
                batch_size=args.tracker_batch_size,
                display=args.display,
                detector_type=args.detector,
                output_format='dict',
                yolo_img_size=args.yolo_img_size,
            )
            # Run multi-object tracker on image folder and store results
            tracking_results = mot(image_folder)


    # remove tracklets if num_frames is less than MIN_NUM_FRAMES
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
            del tracking_results[person_id]

    # Define model 
    if args.regressor == 'hmr-spin':
        model = hmr(path_config.SMPL_MEAN_PARAMS).to(device)
    elif args.regressor == 'emo_body_lang':
        model = emo_body_lang(path_config.SMPL_MEAN_PARAMS, pretrained=True).to(device)

    # Load pretrained weights
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'], strict=True)

    model.eval()
    print(f'Loaded pretrained weights from \"{args.checkpoint}\"')

    # Run pred on each person
    if args.recon_result_file:
        # Load precomputed results if provided
        pred_results = joblib.load(args.recon_result_file)
        print('Loaded results from ' + args.recon_result_file)
    else:
        if args.pre_load_imgs:
            # If pre-loading images is enabled, load and preprocess images
            image_file_names = [
                osp.join(image_folder, x)
                for x in os.listdir(image_folder)
                if x.endswith('.png') or x.endswith('.jpg')
            ]
            image_file_names = sorted(image_file_names)
            image_file_names = np.array(image_file_names)
            pre_load_imgs = []
            for file_name in image_file_names:
                # Read and convert images to RGB format
                pre_load_imgs.append(cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB))
            pre_load_imgs = np.array(pre_load_imgs)
            print('image_file_names', pre_load_imgs.shape)
        else:
            # If not pre-loading images, set image_file_names to None
            image_file_names = None
        
        print(f'Running reconstruction on each tracklet...')
        pred_time = time.time()
        pred_results = {}
        
        # Iterate over each person (tracklet) in tracking_results
        for person_id in tqdm(list(tracking_results.keys())):
            bboxes = joints2d = None

            # Determine if using bbox or pose-based tracking method
            if args.tracking_method == 'bbox':
                bboxes = tracking_results[person_id]['bbox']
            elif args.tracking_method == 'pose':
                joints2d = tracking_results[person_id]['joints2d']

            frames = tracking_results[person_id]['frames']

            if args.pre_load_imgs:
                # Use pre-loaded images for inference if enabled
                print('image_file_names frames', pre_load_imgs[frames].shape)
                dataset = Inference(
                    image_folder=image_folder,
                    frames=frames,
                    bboxes=bboxes,
                    joints2d=joints2d,
                    scale=bbox_scale,
                    pre_load_imgs=pre_load_imgs[frames]
                )
            else:
                # Initialize Inference dataset without pre-loaded images
                dataset = Inference(
                    image_folder=image_folder,
                    frames=frames,
                    bboxes=bboxes,
                    joints2d=joints2d,
                    scale=bbox_scale,
                )

            if args.image_based:
                # Get image shape to compute original height and width
                img_shape = cv2.imread(osp.join(image_folder, os.listdir(image_folder)[frames[0]])).shape
                orig_height, orig_width = img_shape[:2]

            bboxes = dataset.bboxes
            frames = dataset.frames
            has_keypoints = True if joints2d is not None else False

            # Create DataLoader for the current dataset
            dataloader = DataLoader(dataset, batch_size=args.model_batch_size, num_workers=8)

            with torch.no_grad():
                # Initialize lists to store prediction results
                pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []

                for batch in dataloader:
                    if has_keypoints:
                        # Separate joint coordinates if keypoints are available
                        batch, nj2d = batch
                        norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                    # Move batch to specified device (e.g., GPU)
                    batch = batch.to(device)

                    batch_size = batch.shape[0]
                    seqlen = 1
                    
                    # Depending on the selected regressor, perform predictions
                    if args.regressor == 'hmr-spin':
                        # Placeholder for unimplemented regressor
                        raise NotImplementedError()
                    elif args.regressor == 'emo_body_lang':
                        preds_dict, _ = model(batch)

                    # Extract output from prediction dictionary
                    output = preds_dict['smpl_out'][-1]

                    # Append predicted values to respective lists
                    pred_cam.append(output['theta'][:, :3].reshape(batch_size * seqlen, -1))
                    pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                    pred_pose.append(output['theta'][:, 13:85].reshape(batch_size * seqlen, -1))
                    pred_betas.append(output['theta'][:, 3:13].reshape(batch_size * seqlen, -1))
                    pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))

                # Concatenate lists to form final tensors
                pred_cam = torch.cat(pred_cam, dim=0)
                pred_verts = torch.cat(pred_verts, dim=0)
                pred_pose = torch.cat(pred_pose, dim=0)
                pred_betas = torch.cat(pred_betas, dim=0)
                pred_joints3d = torch.cat(pred_joints3d, dim=0)

                del batch  # Clear batch variable from memory

            # Convert tensors to numpy arrays
            pred_cam = pred_cam.cpu().numpy()
            pred_verts = pred_verts.cpu().numpy()
            pred_pose = pred_pose.cpu().numpy()
            pred_betas = pred_betas.cpu().numpy()
            pred_joints3d = pred_joints3d.cpu().numpy()

            # Convert cropped camera coordinates to original image coordinates
            orig_cam = convert_crop_cam_to_orig_img(
                cam=pred_cam,
                bbox=bboxes,
                img_width=orig_width,
                img_height=orig_height
            )

            # Create output dictionary for current person's predictions
            output_dict = {
                'pred_cam': pred_cam,
                'orig_cam': orig_cam,
                'verts': pred_verts,
                'pose': pred_pose,
                'betas': pred_betas,
                'joints3d': pred_joints3d,
                'joints2d': joints2d,
                'bboxes': bboxes,
                'frame_ids': frames,
            }

            # Store output dictionary in pred_results under current person_id
            pred_results[person_id] = output_dict

        del model  # Delete model instance to free up memory

        # Calculate FPS and total processing time
        end = time.time()
        fps = num_frames / (end - pred_time)

        print(f'FPS: {fps:.2f}')
        total_time = time.time() - total_time
        print(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
        print(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')

        # Save prediction results to a pickle file
        print(f'Saving output results to \"{os.path.join(output_path, "output.pkl")}\".')
        joblib.dump(pred_results, os.path.join(output_path, "output.pkl"))


    if not args.no_render:
        # ========= Render results as a single video ========= #
        # Initialize renderer
        renderer = PyRenderer(resolution=(orig_width, orig_height))

        # Create output folder for rendered images
        output_img_folder = os.path.join(output_path, osp.split(image_folder)[-1] + '_output')
        os.makedirs(output_img_folder, exist_ok=True)

        print(f'Rendering output video, writing frames to {output_img_folder}')

        # Prepare results for rendering
        frame_results = prepare_rendering_results(pred_results, num_frames)

        # Get list of image file names in the folder
        image_file_names = sorted([
            os.path.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        # Determine color type based on arguments or configuration
        if args.regressor == 'hmr-spin':
            color_type = 'pink'
        elif cfg.MODEL.EBL.N_ITER == 0 and not cfg.MODEL.EBL.AUX_SUPV_ON:
            color_type = 'neutral'
        else:
            color_type = 'purple'

        # Iterate over each frame
        for frame_idx in tqdm(range(len(image_file_names))):
            img_fname = image_file_names[frame_idx]
            img = cv2.imread(img_fname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize image if specified by render_ratio argument
            if args.render_ratio != 1:
                img = resize(img, (int(img.shape[1] * args.render_ratio), int(img.shape[0] * args.render_ratio)), anti_aliasing=True)
                img = (img * 255).astype(np.uint8)

            raw_img = img.copy()

            # Initialize side view and empty background images if specified
            if args.sideview:
                side_img = np.zeros_like(img)
            
            if args.empty_bg:
                empty_img = np.zeros_like(img)

            # Iterate over each person detected in the frame
            for person_id, person_data in frame_results[frame_idx].items():
                frame_verts = person_data['verts']
                frame_cam = person_data['cam']

                mesh_filename = None

                # Save mesh as .obj file if specified
                if args.save_obj:
                    mesh_folder = os.path.join(output_path, 'meshes', f'{person_id:04d}')
                    os.makedirs(mesh_folder, exist_ok=True)
                    mesh_filename = os.path.join(mesh_folder, f'{frame_idx:06d}.obj')

                # Render image with or without empty background
                if args.empty_bg:
                    img, empty_img = renderer(
                        frame_verts,
                        img=[img, empty_img],
                        cam=frame_cam,
                        color_type=color_type,
                        mesh_filename=mesh_filename
                    )
                else:
                    img = renderer(
                        frame_verts,
                        img=img,
                        cam=frame_cam,
                        color_type=color_type,
                        mesh_filename=mesh_filename
                    )

                # Render side view if specified
                if args.sideview:
                    side_img = renderer(
                        frame_verts,
                        img=side_img,
                        cam=frame_cam,
                        color_type=color_type,
                        angle=270,
                        axis=[0, 1, 0],
                    )

            # Concatenate raw image with rendered image if specified
            if args.with_raw:
                img = np.concatenate([raw_img, img], axis=1)

            # Concatenate empty background image with rendered image if specified
            if args.empty_bg:
                img = np.concatenate([img, empty_img], axis=1)

            # Concatenate side view image with rendered image if specified
            if args.sideview:
                img = np.concatenate([img, side_img], axis=1)

            # Save rendered image to output folder
            cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img)

            # Optionally save rendered image with original image file name
            if args.image_based:
                imsave(os.path.join(output_img_folder, osp.split(img_fname)[-1][:-4]+'.png'), img)
            else:
                imsave(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img)

            # Display rendered image if specified
            if args.display:
                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Close display window if opened
        if args.display:
            cv2.destroyAllWindows()

        # Save rendered video 
        # Determine video name based on input folder or file name
        vid_name = osp.split(image_folder)[-1] if args.image_folder is not None else os.path.basename(video_file)
        save_name = f'{vid_name.replace(".mp4", "")}_result.mp4'
        save_name = os.path.join(output_path, save_name)

        # Save rendered images as video if not image-based rendering
        if not args.image_based:
            print(f'Saving result video to {save_name}')
            images_to_video(img_folder=output_img_folder, output_vid_file=save_name)

    # Clean up output and image folders
    # shutil.rmtree(output_img_folder)
    # shutil.rmtree(image_folder)

    print('================= DONE =================')



if __name__ == '__main__':
    """
    Parse and validate command-line arguments.
    Args:
        args (argparse.Namespace): Arguments from command line.
    Returns:
        argparse.Namespace: Parsed and validated arguments.
    """
    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # Define command line arguments
    parser.add_argument('--img_file', type=str,
                        help='Path to a single input image')
    parser.add_argument('--vid_file', type=str,
                        help='Input video path or YouTube link')
    parser.add_argument('--image_folder', type=str, default=None,
                        help='Input image folder')
    parser.add_argument('--output_folder', type=str, default='output',
                        help='Output folder to write results')
    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'],
                        help='Tracking method to calculate the tracklet of a subject from the input video')
    parser.add_argument('--detector', type=str, default='maskrcnn', choices=['yolo', 'maskrcnn'],
                        help='Object detector to be used for bbox tracking')
    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='Input image size for YOLO detector')
    parser.add_argument('--tracker_batch_size', type=int, default=12,
                        help='Batch size of object detector used for bbox tracking')
    parser.add_argument('--staf_dir', type=str, default='/home/abbas/PMRR/STAF/',
                        help='Path to directory of STAF pose tracking method')
    parser.add_argument('--regressor', type=str, default='emo_body_lang', 
                        help='Name of the SMPL regressor')
    parser.add_argument('--cfg_file', type=str, default='configs/emo_pose.yaml',
                        help='Config file path')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to network checkpoint')
    parser.add_argument('--misc', default=None, type=str, nargs="*",
                        help='Other parameters')
    parser.add_argument('--model_batch_size', type=int, default=8,
                        help='Batch size for SMPL prediction')
    parser.add_argument('--display', action='store_true',
                        help='Visualize the results of each step during demo')
    parser.add_argument('--no_render', action='store_true',
                        help='Disable final rendering of output video')
    parser.add_argument('--with_raw', action='store_true',
                        help='Attach raw image')
    parser.add_argument('--empty_bg', action='store_true',
                        help='Render meshes on empty background')
    parser.add_argument('--sideview', action='store_true',
                        help='Render meshes from alternate viewpoint')
    parser.add_argument('--image_based', action='store_true',
                        help='Image-based reconstruction')
    parser.add_argument('--use_gt', action='store_true',
                        help='Use the ground truth tracking annotations')
    parser.add_argument('--anno_file', type=str, default='',
                        help='Path to tracking annotation file')
    parser.add_argument('--render_ratio', type=float, default=1.,
                        help='Ratio for render resolution')
    parser.add_argument('--recon_result_file', type=str, default='',
                        help='Path to reconstruction result file')
    parser.add_argument('--pre_load_imgs', action='store_true',
                        help='Pre-load input images')
    parser.add_argument('--save_obj', action='store_true',
                        help='Save results as .obj files')

    # Parse arguments from command line
    args = parser.parse_args()
    parse_args(args)  # Validate and process arguments

    # Decide whether to run image demo or video demo based on provided arguments
    if args.img_file is not None:
        print('Run demo for a single input image.')
        run_image_demo(args)  # Call function to process single image demo
    else:
        print('Run demo for a video input.')
        run_video_demo(args)  # Call function to process video demo
