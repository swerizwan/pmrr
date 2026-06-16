from os.path import join, expanduser

# Root directories for various datasets
H36M_ROOT = join('/home/abbas/EmoBodyLang/datasets/preprocess', 'Datasets/h36m')  # Human3.6M dataset root
LSP_ROOT = join('/home/abbas/EmoBodyLang/datasets/preprocess', 'Datasets/human/LSP/lsp_dataset_small')  # LSP dataset root
LSP_ORIGINAL_ROOT = join('/home/abbas/EmoBodyLang/datasets/preprocess', 'Datasets/LSP/lsp_dataset_original')  # Original LSP dataset root
LSPET_ROOT = join('/home/abbas/EmoBodyLang/datasets/preprocess', 'Datasets/LSP/hr-lspet')  # LSPET dataset root
MPII_ROOT = join('/home/abbas/EmoBodyLang/datasets/preprocess', 'Datasets/mpii')  # MPII Human Pose dataset root
COCO_ROOT = join('/home/abbas/EmoBodyLang/datasets/preprocess', 'coco')  # COCO dataset root
MPI_INF_3DHP_ROOT = join('/home/abbas/EmoBodyLang/datasets/preprocess',
                         'Datasets/mpi_inf_3dhp_train_set')  # MPI-INF-3DHP dataset root
PW3D_ROOT = join('/home/abbas/EmoBodyLang/datasets/preprocess', '3dpw')  # 3DPW dataset root
UPI_S1H_ROOT = join('/home/abbas/EmoBodyLang/datasets/preprocess', 'Datasets/human/upi-s1h')  # UPI-S1H dataset root
SURREAL_ROOT = join('/home/abbas/EmoBodyLang/datasets/preprocess', 'Datasets/human/SURREAL/data')  # SURREAL dataset root
threeDOH50K_ROOT = join('/home/abbas/EmoBodyLang/datasets/preprocess', 'Datasets/human/3DOH50K')  # 3DOH50K dataset root

# Output folder to save test/train npz files
DATASET_NPZ_PATH = 'data/dataset_extras'

OPENPOSE_PATH = 'datasets/openpose'  # OpenPose dataset path

# Dictionary mapping dataset keys to their respective root directories
DATASET_FOLDERS = {
    'h36m': H36M_ROOT,
    'h36m-p1': H36M_ROOT,
    'h36m-p2': H36M_ROOT,
    'h36m-p2-mosh': H36M_ROOT,
    'lsp-orig': LSP_ORIGINAL_ROOT,
    'lsp': LSP_ROOT,
    'lspet': LSPET_ROOT,
    'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
    'mpii': MPII_ROOT,
    'coco': COCO_ROOT,
    'dp_coco': COCO_ROOT,
    '3dpw': PW3D_ROOT,
    'upi-s1h': UPI_S1H_ROOT,
    'surreal': SURREAL_ROOT,
    '3doh50k': threeDOH50K_ROOT,
    'coco-full': COCO_ROOT
}

# List of dictionaries containing paths to npz files for different datasets and subsets
DATASET_FILES = [{
    'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_valid_protocol1.npz'),
    'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_valid_protocol2_newpath.npz'),
    'h36m-p2-mosh': join(DATASET_NPZ_PATH, 'h36m_mosh_valid_p2.npz'),
    'lsp': join(DATASET_NPZ_PATH, 'lsp_dataset_test.npz'),
    'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_test.npz'),
    # 'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_valid.npz'),
    '3dpw': join(DATASET_NPZ_PATH, '3dpw_test.npz'),
    'coco': join(DATASET_NPZ_PATH, 'coco_2014_val.npz'),
    'dp_coco': join(DATASET_NPZ_PATH, 'dp_coco_2014_minival.npz'),
    'surreal': join(DATASET_NPZ_PATH, 'surreal_val.npz'),
    '3doh50k': join(DATASET_NPZ_PATH, 'threeDOH50K_testset.npz'),
    'agora': join(DATASET_NPZ_PATH, 'agora_test.npz'),
    'freihand': join(DATASET_NPZ_PATH, 'freihand_evaluation.npz'),
},
{
    'h36m': join(DATASET_NPZ_PATH, 'h36m_mosh_train.npz'),
    '3dpw': join(DATASET_NPZ_PATH, '3dpw_train.npz'),
    'lsp-orig': join(DATASET_NPZ_PATH, 'lsp_dataset_original_train.npz'),
    'mpii': join(DATASET_NPZ_PATH, 'mpii_train_eft.npz'),
    'coco': join(DATASET_NPZ_PATH, 'coco_2014_train_eft.npz'),
    'coco-full': join(DATASET_NPZ_PATH, 'coco-full_train_eft.npz'),
    'coco-hf': join(DATASET_NPZ_PATH, 'coco-hf_train_eft.npz'),
    'coco-hf-x': join(DATASET_NPZ_PATH, 'coco-hf_train_smplx.npz'),
    'dp_coco': join(DATASET_NPZ_PATH, 'dp_coco_2014_train.npz'),
    'lspet': join(DATASET_NPZ_PATH, 'hr-lspet_train_eft.npz'),
    'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train.npz'),
    'surreal': join(DATASET_NPZ_PATH, 'surreal_train.npz'),
    '3doh50k': join(DATASET_NPZ_PATH, 'threeDOH50K_trainset.npz'),
    'agora': join(DATASET_NPZ_PATH, 'agora_train.npz'),  # Should be Training!!!
    'freihand': join(DATASET_NPZ_PATH, 'freihand_training.npz'),
}]

# Paths to various files used in the project
CUBE_PARTS_FILE = 'data/cube_parts.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'
VERTEX_TEXTURE_FILE = 'data/vertex_texture.npy'
STATIC_FITS_DIR = 'data/static_fits'
FINAL_FITS_DIR = 'data/final_fits'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/smpl'
