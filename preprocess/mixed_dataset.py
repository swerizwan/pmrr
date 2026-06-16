import torch
import numpy as np

from .main_dataset import BaseDataset


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, options, **kwargs):
        # List of dataset names to be used
        self.dataset_list = ['h36m', 'lsp-orig', 'mpii', 'lspet', 'coco-full', 'mpi-inf-3dhp']
        
        # Mapping of dataset names to their indices
        self.dataset_dict = {'h36m': 0, 'lsp-orig': 1, 'mpii': 2, 'lspet': 3, 'coco-full': 4, 'mpi-inf-3dhp': 5}

        # Initialize each dataset using the BaseDataset class and store in a list
        self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
        
        # Calculate the length of each dataset and store in a dictionary
        self.dataset_length = {self.dataset_list[idx]: len(ds) for idx, ds in enumerate(self.datasets)}
        
        # Calculate the total length of ITW (in-the-wild) datasets (all except first and last)
        length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
        
        # Determine the maximum length among all datasets to set the length of the MixedDataset
        self.length = max([len(ds) for ds in self.datasets])
        
        """
        Data distribution inside each batch:
        30% H36M - 60% ITW - 10% MPI-INF
        """
        # Define the partition ratios for each dataset
        self.partition = [
            .5,  # 50% of the data from the first dataset (H36M)
            .3 * len(self.datasets[1]) / length_itw,  # 30% of ITW data from the second dataset (lsp-orig)
            .3 * len(self.datasets[2]) / length_itw,  # 30% of ITW data from the third dataset (mpii)
            .3 * len(self.datasets[3]) / length_itw,  # 30% of ITW data from the fourth dataset (lspet)
            .3 * len(self.datasets[4]) / length_itw,  # 30% of ITW data from the fifth dataset (coco-full)
            0.2   # 20% of the data from the last dataset (mpi-inf-3dhp)
        ]
        
        # Convert partition list to a cumulative sum array
        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        # Generate a random number between 0 and 1
        p = np.random.rand()
        
        # Determine which dataset to sample from based on the random number
        for i in range(6):
            if p <= self.partition[i]:
                # Return the item from the selected dataset, using modulo to handle index overflow
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        # Return the length of the longest dataset
        return self.length
