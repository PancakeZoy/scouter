import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import anndata as ad
import numpy as np
import pickle
import random
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from _loss import *

def gears_loss(predicted, true, control, group, gamma=2.0, lambda_=0.1):
    # Autofocus loss calculation
    DiffExpr_CellAvg = ((true - predicted).abs() ** (2 + gamma)).mean(axis=1)
    autofocus_loss = group_mean(DiffExpr_CellAvg, group).mean()
    
    # Direction-aware loss calculation    
    DiffSign_CellAvg = ((torch.sign(true - control) - torch.sign(predicted - control))**2).mean(axis=1)
    direction_aware_loss = group_mean(DiffSign_CellAvg, group).mean()

    # Total loss
    total_loss = autofocus_loss + lambda_ * direction_aware_loss
    return total_loss

def group_mean(expr, group):
    unique_ids, inverse_indices = torch.unique(group, return_inverse=True)
    num_groups = unique_ids.size(0)
    group_sums = torch.zeros(num_groups)
    group_sums.index_add_(0, inverse_indices, expr)
    group_counts = torch.zeros(num_groups).long().index_add_(0, inverse_indices, torch.ones_like(group))
    group_means = group_sums / group_counts
    return group_means

# def split_data(adata, 
#                test_conds = None, 
#                val_ratio: float=0.2):
#     """Splits the data into train, validation, and test sets based on conditions."""
#     if test_conds is None:
#         all_conds = adata[adata.obs['control'] == 0].obs['condition'].unique().tolist()
#         test_conds = random.sample(all_conds, int(len(all_conds)*0.1))

#     test_mask = adata.obs['condition'].isin(test_conds)
#     train_val_mask = ~test_mask

#     train_val_adata = adata[train_val_mask]
#     test_adata = adata[test_mask]

#     train_val_indices = np.arange(len(train_val_adata))
#     train_indices, val_indices = train_test_split(train_val_indices, test_size=val_ratio, stratify=train_val_adata.obs['condition'])

#     train_adata = train_val_adata[train_indices]
#     val_adata = train_val_adata[val_indices]

#     return train_adata, val_adata, test_adata

def split_data(adata, 
                test_conds = None, 
                val_ratio: float=0.2):
    """Splits the data into train, validation, and test sets based on conditions."""
    if test_conds is None:
        all_conds = adata[adata.obs['control'] == 0].obs['condition'].unique().tolist()
        np.random.shuffle(all_conds)
        split_indices = [int(0.8 * len(all_conds)), int(0.9 * len(all_conds))]
        train_conds, val_conds, test_conds = np.split(all_conds, split_indices)

    train_mask = adata.obs['condition'].isin(list(train_conds)+['ctrl'])
    val_mask = adata.obs['condition'].isin(list(val_conds)+['ctrl'])
    test_mask = adata.obs['condition'].isin(list(test_conds)+['ctrl'])

    train_adata = adata[train_mask]
    val_adata = adata[val_mask]
    test_adata = adata[test_mask]

    return train_adata, val_adata, test_adata


class BalancedDataset(Dataset):
    def __init__(self, 
                 adata, 
                 method: str='both',
                 balance: bool=False,
                 shuffle: bool=True):
        
        self.ctrl_exp = adata[adata.obs['control'] == 1].X.toarray()

        self.pert_adata = adata[adata.obs['control'] == 0]
        self.pert_exp = self.pert_adata.X.toarray()
        self.gene_idx = self.pert_adata.obs['gene_index'].values
        self.conditions = self.pert_adata.obs['condition'].values
        self.bcode = self.pert_adata.obs.index.tolist()
        
        if balance:
            self.PertCellIndx = self._GetPertIndx(method)
        else:
            self.PertCellIndx = np.arange(len(self.pert_exp))
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
        pert_expr = torch.tensor(self.pert_exp[self.PertCellIndx[idx]], dtype=torch.float32)
        gene_idx = torch.tensor(self.gene_idx[self.PertCellIndx[idx]], dtype=torch.long)
        bcode = self.bcode[self.PertCellIndx[idx]]
        return ctrl_exp, pert_expr, gene_idx, bcode
        
    def _GetCtrlIndx(self, num_samples):
        num_control_cells = len(self.ctrl_exp)
        indices = np.arange(num_control_cells)
        sampled_indices = np.resize(indices, num_samples)
        return sampled_indices
    
    def _GetPertIndx(self, method):
        perturbed_data = pd.DataFrame({
            'index': np.arange(len(self.pert_exp)),
            'condition': self.conditions
        })
        
        balanced_indices = []
        unique_conds = perturbed_data['condition'].unique()
        
        if method == 'downsample':
            min_count = perturbed_data['condition'].value_counts().min()
            for cond in unique_conds:
                subset = perturbed_data[perturbed_data['condition'] == cond]
                balanced_subset = subset.sample(min_count)
                balanced_indices.extend(balanced_subset['index'].tolist())
                
        elif method == 'upsample':
            max_count = perturbed_data['condition'].value_counts().max()
            for cond in unique_conds:
                subset = perturbed_data[perturbed_data['condition'] == cond]
                balanced_subset = resample(subset, replace=True, n_samples=max_count)
                balanced_indices.extend(balanced_subset['index'].tolist())
                
        elif method == 'both':
            target_count = len(self.pert_exp) // len(unique_conds)
            for cond in unique_conds:
                subset = perturbed_data[perturbed_data['condition'] == cond]
                if len(subset) >= target_count:
                    balanced_subset = subset.sample(target_count)
                else:
                    balanced_subset = resample(subset, replace=True, n_samples=target_count)
                balanced_indices.extend(balanced_subset['index'].tolist())
        
        return balanced_indices


class GenePerturbationModel(nn.Module):
    def __init__(self, 
                 n_genes: int, 
                 gene_embd, 
                 n_hidden_encoder: tuple=(2048, 512), 
                 n_out_encoder: int=64, 
                 n_hidden_generator: tuple=(2048, 3072),
                 use_batch_norm: bool=True, 
                 use_layer_norm: bool=False,
                 dropout_rate: float = 0.):
        
        """
        Initialize the GenePerturbationModel.

        Parameters:
        - n_genes: 
            Number of input genes.
        - gene_embd: 
            Gene embedding matrix.
        - n_hidden_encoder: 
            Tuple specifying the hidden layer sizes for the cell encoder.
        - n_out_encoder: 
            Size of the output layer for the cell encoder.
        - n_hidden_generator: 
            Tuple specifying the hidden layer sizes for the generator.
        - use_batch_norm: 
            Whether to use batch normalization.
        - use_layer_norm: 
            Whether to use layer normalization.
        - dropout_rate: 
            Dropout rate.
        """
        
        super(GenePerturbationModel, self).__init__()
        self.gene_embd = nn.Parameter(torch.tensor(gene_embd, dtype=torch.float32), requires_grad=False)
        Gembd_dim = self.gene_embd.shape[1]
        self.encoder = self._build_mlp(n_genes, 
                                       n_hidden_encoder, 
                                       n_out_encoder, 
                                       use_batch_norm, 
                                       use_layer_norm, 
                                       dropout_rate)
        self.generator = self._build_mlp(n_out_encoder + Gembd_dim, 
                                         n_hidden_generator, 
                                         n_genes, 
                                         use_batch_norm, 
                                         use_layer_norm, 
                                         dropout_rate)

    def forward(self, gene_idx, ctrl_exp):
        inp_1 = self.gene_embd[gene_idx]
        P_small = self.encoder(ctrl_exp)
        concatenated_input = torch.cat((inp_1, P_small), dim=-1)
        output = self.generator(concatenated_input)
        return output
        
    def _build_mlp(self, 
                   input_dim, 
                   hidden_dims, 
                   output_dim, 
                   use_batch_norm, 
                   use_layer_norm, 
                   dropout_rate):
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            if use_batch_norm:
                # layers.append(nn.BatchNorm1d(h_dim, momentum=0.01, eps=0.001))
                layers.append(nn.BatchNorm1d(h_dim))
            if use_layer_norm:
                # layers.append(nn.LayerNorm(h_dim, elementwise_affine=False))
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.SELU())
            if dropout_rate > 0.:
                layers.append(nn.AlphaDropout(dropout_rate))
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

# Example usage
adata = ad.read_h5ad('/Users/pancake/Downloads/PerturbGPT/PerturbData/adamson/perturb_processed.h5ad')
adata.obs['condition'] = adata.obs['condition'].str.replace(r'\+ctrl$', '', regex=True)
gene_name = adata.var['gene_name'].tolist()

embd_path = '/Users/pancake/Downloads/PerturbGPT/PerturbData/GeneEmb/GenePT_emb/GenePT_gene_embedding_ada_text.pickle'
with open(embd_path, 'rb') as f:
    gene_embd = pd.DataFrame(pickle.load(f)).T
    embd_name = gene_embd.index.tolist()
    match_name = sorted(list(set(embd_name) & set(gene_name)))
    gene_embd = gene_embd.loc[match_name].values

perturb = adata.obs['condition'].unique().tolist()
perturb_indx = {g: (match_name.index(g) if g in match_name else -1) for g in perturb}
adata.obs['gene_index'] = adata.obs['condition'].map(perturb_indx).astype(int)

# Split the data into training, validation, and test sets
train_adata, val_adata, test_adata = split_data(adata)

# Initialize model parameters
n_genes = adata.shape[1]; batch_size = 128

# Initialize model
model = GenePerturbationModel(n_genes, gene_embd)

# Define loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Prepare data loaders
train_dataset = BalancedDataset(train_adata)
val_dataset = BalancedDataset(val_adata)
test_dataset = BalancedDataset(test_adata)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Training and validation loop
num_epochs = 20
for epoch in tqdm(range(num_epochs), desc='Training', unit='epoch'):
    
    # Training phase
    model.train()
    train_loss = 0.0
    # train_loader = tqdm(train_loader, desc='Training Batch', unit='batch')
    for ctrl_exp, pert_expr, gene_idx, bcode in train_loader:
        optimizer.zero_grad()
        output = model(gene_idx, ctrl_exp)
        loss = gears_loss(output, pert_expr, ctrl_exp, gene_idx)
        # print(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        # val_loader = tqdm(val_loader, desc='Validation Batch', unit='batch')
        for ctrl_exp, pert_expr, gene_idx, bcode in val_loader:
            output = model(gene_idx, ctrl_exp)
            loss = gears_loss(output, pert_expr, ctrl_exp, gene_idx)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')



