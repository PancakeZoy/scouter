import torch
import numpy as np

def gears_loss(pred_expr, true_expr, ctrl_expr, group, 
               nonzero_idx_dict, gamma, lambda_):

    unique_cond, idx = np.unique(group, return_inverse=True)
    idx_dict = {val: [] for val in unique_cond}
    for i, cond in enumerate(idx):
        idx_dict[unique_cond[cond]].append(i)
    
    total_loss = torch.tensor(0.0, device=pred_expr.device)
    for p in set(group):
        cell_idx = idx_dict[p]
        retain_gene_idx = list(nonzero_idx_dict[p])
        
        p_pred = pred_expr[cell_idx][:,retain_gene_idx]
        p_true = true_expr[cell_idx][:,retain_gene_idx]
        p_ctrl = ctrl_expr[cell_idx][:,retain_gene_idx]
        
        # Autofocus loss calculation
        autofocus_loss = ((p_true - p_pred).abs() ** (2 + gamma)).mean()
        # Direction-aware loss calculation
        direction_aware_loss = ((torch.sign(p_true - p_ctrl) - torch.sign(p_pred - p_ctrl))**2).mean()
        # Total loss
        total_loss += (autofocus_loss + lambda_ * direction_aware_loss)
    
    return total_loss/(len(set(group)))


def split_TrainVal(adata, key_label, val_conds_include, val_ratio, seed):
    all_conds = adata.obs[key_label].unique().tolist()
    all_conds.remove('ctrl')
    if val_conds_include is None:
        np.random.seed(seed)
        np.random.shuffle(all_conds)
        n_ValNeed = round(val_ratio * len(all_conds))        
        val_conds, train_conds = np.split(all_conds, [n_ValNeed])
        train_conds = list(train_conds)+['ctrl']
        val_conds = list(val_conds)+['ctrl']
    else:
        val_conds = list(val_conds_include)+['ctrl']
        train_conds = list(np.setdiff1d(all_conds, val_conds))+['ctrl']
        
    train_mask = adata.obs['condition'].isin(train_conds)
    val_mask = adata.obs['condition'].isin(val_conds)
    train_adata = adata[train_mask]
    val_adata = adata[val_mask]

    return train_conds, train_adata, val_conds, val_adata
