import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.utils import resample

class BalancedDataset(Dataset):
    def __init__(self, 
                 adata, 
                 key_label: str,
                 key_embd_index: str,
                 balance: bool=False,                 
                 balance_method: str='both',
                 shuffle: bool=True):
        
        """
        Initialize the Dataset where the number of control cells matches with that of perturbed cells.
    
        Parameters:
        - adata: 
            The adata object that contains all cells (e.g. control, and perturbed)
        - balance: 
            Whether to balance the perturbed samples. Default is False, since the loss function only considers the mean error for each perturbation.
        - balance_method: 
            The approach to balance the perturbed samples if 'balance' = True. Options are:
                'downsample': Downsamples all the perturbed conditions to the minimum number of samples across conditions.
                'upsample': Upsamples all the perturbed conditions to the maximum number of samples across conditions.
                'both': Balances the number of samples across conditions by either upsampling or downsampling to a target number of samples, which is the total number of perturbed cells divided by total number of perturbations.
        - shuffle: 
            Whether to shuffle control cells and perturbed cells. Default is to True.
        """
        
        self.ctrl_exp = adata[adata.obs[key_label] == 'ctrl'].X.toarray()

        self.pert_adata = adata[adata.obs[key_label] != 'ctrl']
        self.X = self.pert_adata.X.toarray()
        self.gene_idx = self.pert_adata.obs[key_embd_index].values
        self.conditions = self.pert_adata.obs[key_label].values
        self.bcode = self.pert_adata.obs.index.tolist()
        
        if balance:
            self.PertCellIndx = self._GetPertIndx(balance_method)
        else:
            self.PertCellIndx = np.arange(len(self.X))
        self.CtrlCellIndx = self._GetCtrlIndx(len(self.PertCellIndx))
        
        if shuffle:
            np.random.shuffle(self.PertCellIndx)
            np.random.shuffle(self.CtrlCellIndx)
            
        if len(self.PertCellIndx) != len(self.CtrlCellIndx):
            raise ValueError("Number of Perturbed cells and Control cells must be equal.")
        
    def __len__(self):
        return len(self.PertCellIndx)
    
    def __getitem__(self, idx):
        ctrl_exp = torch.tensor(self.ctrl_exp[self.CtrlCellIndx[idx]], dtype=torch.float32)
        X = torch.tensor(self.X[self.PertCellIndx[idx]], dtype=torch.float32)
        gene_idx = torch.tensor(self.gene_idx[self.PertCellIndx[idx]], dtype=torch.long)
        bcode = self.bcode[self.PertCellIndx[idx]]
        return ctrl_exp, X, gene_idx, bcode
        
    def _GetCtrlIndx(self, num_samples):
        num_control_cells = len(self.ctrl_exp)
        indices = np.arange(num_control_cells)
        sampled_indices = np.resize(indices, num_samples)
        return sampled_indices
    
    def _GetPertIndx(self, balance_method):
        perturbed_data = pd.DataFrame({
            'index': np.arange(len(self.X)),
            'condition': self.conditions
        })
        valid_methods = ['both', 'downsample', 'upsample']
        if balance_method not in valid_methods:
            raise ValueError(f"Invalid balance_method: {balance_method}. Must be one of {valid_methods}")  
            
        balanced_indices = []
        unique_conds = perturbed_data['condition'].unique()
        
        if balance_method == 'downsample':
            min_count = perturbed_data['condition'].value_counts().min()
            for cond in unique_conds:
                subset = perturbed_data[perturbed_data['condition'] == cond]
                balanced_subset = subset.sample(min_count)
                balanced_indices.extend(balanced_subset['index'].tolist())
                
        elif balance_method == 'upsample':
            max_count = perturbed_data['condition'].value_counts().max()
            for cond in unique_conds:
                subset = perturbed_data[perturbed_data['condition'] == cond]
                balanced_subset = resample(subset, replace=True, n_samples=max_count)
                balanced_indices.extend(balanced_subset['index'].tolist())
                
        elif balance_method == 'both':
            target_count = len(self.X) // len(unique_conds)
            for cond in unique_conds:
                subset = perturbed_data[perturbed_data['condition'] == cond]
                if len(subset) >= target_count:
                    balanced_subset = subset.sample(target_count)
                else:
                    balanced_subset = resample(subset, replace=True, n_samples=target_count)
                balanced_indices.extend(balanced_subset['index'].tolist())
        
        return balanced_indices