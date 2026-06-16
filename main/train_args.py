import argparse

class TrainOptions():
    def __init__(self):
        # Initialize an argument parser object
        self.parser = argparse.ArgumentParser()

        # General options group
        gen = self.parser.add_argument_group('General')
        # Add a boolean flag for resuming training from a checkpoint
        gen.add_argument('--resume', dest='resume', default=False, action='store_true', help='Resume from checkpoint (Use latest checkpoint by default)')

        # Input/Output options group
        io = self.parser.add_argument_group('io')
        # Directory to store logs
        io.add_argument('--log_dir', default='logs', help='Directory to store logs')
        # Path to a pretrained checkpoint to load at the start of training
        io.add_argument('--pretrained_checkpoint', default=None, help='Load a pretrained checkpoint at the beginning of training')

        # Training options group
        train = self.parser.add_argument_group('Training Options')
        # Total number of training epochs
        train.add_argument('--num_epochs', type=int, default=1, help='Total number of training epochs')
        # Name of the SMPL regressor to use
        train.add_argument('--regressor', type=str, choices=['hmr', 'emo_body_lang'], default='emo_body_lang', help='Name of the SMPL regressor.')
        # Path to the configuration file
        train.add_argument('--cfg_file', type=str, default='./configs/emo_pose.yaml', help='Config file path for PMRR.')
        # Image resolution to which bounding boxes will be rescaled
        train.add_argument('--img_res', type=int, default=224, help='Rescale bounding boxes to size [img_res, img_res] before feeding them into the network')
        # Range for random rotation augmentation
        train.add_argument('--rot_factor', type=float, default=30, help='Random rotation in the range [-rot_factor, rot_factor]')
        # Range for random noise augmentation
        train.add_argument('--noise_factor', type=float, default=0.4, help='Randomly multiply pixel values with factor in the range [1-noise_factor, 1+noise_factor]')
        # Range for random scaling augmentation
        train.add_argument('--scale_factor', type=float, default=0.25, help='Rescale bounding boxes by a factor of [1-scale_factor, 1+scale_factor]')
        # Weight for OpenPose keypoints during training
        train.add_argument('--openpose_train_weight', default=0., help='Weight for OpenPose keypoints during training')
        # Weight for ground truth keypoints during training
        train.add_argument('--gt_train_weight', default=1., help='Weight for GT keypoints during training')
        # Name of the dataset to use for evaluation
        train.add_argument('--eval_dataset', type=str, default='3dpw', help='Name of the evaluation dataset.')
        # Flag to use a single dataset for training
        train.add_argument('--single_dataset', default='3dpw', action='store_true', help='Use a single dataset')
        # Name of the single dataset to use
        train.add_argument('--single_dataname', type=str, default='coco-full', help='Name of the single dataset.')
        # Flag to evaluate PVE (Per Vertex Error)
        train.add_argument('--eval_pve', default=False, action='store_true', help='Evaluate PVE')
        # Flag to overwrite the latest checkpoint
        train.add_argument('--overwrite', default=False, action='store_true', help='Overwrite the latest checkpoint')

        # Distributed training options
        train.add_argument('--distributed', action='store_true', help='Use distributed training')
        train.add_argument('--dist_backend', default='nccl', type=str, help='Distributed backend')
        train.add_argument('--dist_url', default='tcp://127.0.0.1:10356', type=str, help='URL used to set up distributed training')
        train.add_argument('--world_size', default=1, type=int, help='Number of nodes for distributed training')
        train.add_argument("--local_rank", default=0, type=int, help='Local rank of the process (used by torch.distributed)')
        train.add_argument('--rank', default=0, type=int, help='Node rank for distributed training')
        train.add_argument('--multiprocessing_distributed', action='store_true', help='Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training')

        # Miscellaneous options group
        misc = self.parser.add_argument_group('Misc Options')
        # Allow modification of config options using the command line
        misc.add_argument('--misc', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    def parse_args(self):
        """Parse input arguments."""
        # Parse the command-line arguments
        self.args = self.parser.parse_args()
        # Save the parsed arguments to a file (currently not implemented)
        self.save_dump()
        return self.args

    def save_dump(self):
        """Store all argument values to a json file.
        The default location is logs/expname/args.json.
        """
        # Functionality to save arguments is not implemented
        pass
