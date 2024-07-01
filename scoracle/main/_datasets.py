import torch
from torch.utils.data import Dataset
import numpy as np
import anndata as ad

class BalancedDataset(Dataset):
    def __init__(self, 
                 adata: ad.AnnData, 
                 key_label: str,
                 key_embd_index: str,
                 shuffle: bool=True,
                 seed: int=24):
        
        """
        Initialize the Dataset where the number of control cells matches with that of perturbed cells.
    
        Parameters:
        adata: 
            The adata object that contains all cells (e.g. control, and perturbed)
        key_label:
            The column name of `adata.obs` that contains the perturbation condition
        key_embd_index:
            The column name of `adata.obs` that contains the index of perturbed genes in the gene embedding matrix
        shuffle: 
            Whether to shuffle control cells and perturbed cells. Default is to True.
        seed : int, optional
            Random seed for reproducibility. Default is 24.
        """
        
        self.ctrl_expr = adata[adata.obs[key_label] == 'ctrl'].X.toarray()
        self.pert_adata = adata[adata.obs[key_label] != 'ctrl']
        self.true_expr = self.pert_adata.X.toarray()
        self.pert_idx = self.pert_adata.obs[key_embd_index].values
        self.conditions = self.pert_adata.obs[key_label].values
        self.bcode = self.pert_adata.obs.index.tolist()
        
        self.n_pert_cells = len(self.true_expr)
        self.n_ctrl_cells = len(self.ctrl_expr)
        self.PertCellIndx = np.arange(self.n_pert_cells)
        self.CtrlCellIndx = self._GetCtrlIndx(self.n_pert_cells)
        
        if shuffle:
            np.random.shuffle(self.PertCellIndx)
            np.random.shuffle(self.CtrlCellIndx)
            
        if len(self.PertCellIndx) != len(self.CtrlCellIndx):
            raise ValueError("Number of Perturbed cells and Control cells must be equal.")
        
    def __len__(self):
        return len(self.PertCellIndx)
    
    def __getitem__(self, idx):
        ctrl_expr = torch.tensor(self.ctrl_expr[self.CtrlCellIndx[idx]], dtype=torch.float32)
        true_expr = torch.tensor(self.true_expr[self.PertCellIndx[idx]], dtype=torch.float32)
        pert_idx = torch.tensor(self.pert_idx[self.PertCellIndx[idx]], dtype=torch.long)
        bcode = self.bcode[self.PertCellIndx[idx]]
        return ctrl_expr, true_expr, pert_idx, bcode
        
    def _GetCtrlIndx(self, num_samples):
        num_control_cells = self.n_ctrl_cells
        indices = np.arange(num_control_cells)
        sampled_indices = np.resize(indices, num_samples)
        return sampled_indices