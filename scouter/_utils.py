import torch
import numpy as np
import os
import gzip
import shutil
import requests
from tqdm import tqdm
import anndata as ad
import pandas as pd

def gears_loss(pred_expr, true_expr, ctrl_expr, group, 
               nonzero_idx_dict, gamma, lambda_):
    
    # Autofocus loss calculation
    autofocus_loss = ((true_expr - pred_expr).abs() ** (2 + gamma))
    # Direction-aware loss calculation
    direction_aware_loss = ((torch.sign(true_expr - ctrl_expr) - torch.sign(pred_expr - ctrl_expr))**2)
    # Total loss
    total_loss = autofocus_loss + lambda_ * direction_aware_loss
    unique_cond, cells_cond_index = np.unique(group, return_inverse=True)
    
    idx_dict = {val: [] for val in unique_cond}
    for idx_cell, idx_cond in enumerate(cells_cond_index):
        cond = unique_cond[idx_cond]
        idx_dict[cond].append(idx_cell)
    
    loss_scalar = torch.tensor(0.0, device=pred_expr.device)
    for p in unique_cond:
        cell_idx = idx_dict[p]
        retain_gene_idx = list(nonzero_idx_dict[p])
        loss_scalar += total_loss[cell_idx][:,retain_gene_idx].mean()
    
    return loss_scalar/(len(unique_cond))


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

def download_and_extract(url, save_dir, filename):
    """
    Download and unzip a file if it's compressed (.gz).

    Parameters
    ----------
    url : str
        URL of the file to download.
    save_dir : str
        Directory where the file will be saved.
    filename : str
        Name of the file to be saved in `save_dir`.

    Returns
    -------
    str
        Path to the downloaded (and unzipped if applicable) file.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)
    uncompressed_path = file_path.rstrip('.gz')
    
    # Check if the uncompressed file already exists
    if not os.path.exists(uncompressed_path):
        # Download the compressed file if it hasn't been downloaded
        if not os.path.exists(file_path):
            print(f'Downloading {filename}...')
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size_in_bytes, unit='B', unit_scale=True, unit_divisor=1024)
            with open(file_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("ERROR: Something went wrong during the download. The downloaded file might be corrupted.")
        
        # Unzip the .gz file and delete it
        if file_path.endswith('.gz'):
            print(f'Unzipping {filename}...')
            with gzip.open(file_path, 'rb') as f_in:
                with open(uncompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(file_path)
            print('Unzipped and removed the compressed file.')
    else:
        print(f'{filename} already available and uncompressed.')
    
    return uncompressed_path


def adamson_small(save_dir='./data'):
    """
    Download, unzip, and load a small version of Adamson dataset as a demo dataset.
    
    Returns
    -------
    anndata.AnnData
        The example dataset loaded into an AnnData object.
    """
    url = 'https://github.com/PancakeZoy/scouter_misc/raw/main/data/Data_Demo/Adamson_small.h5ad.gz'
    dataset_filename = 'Adamson_small.h5ad.gz'
    file_path = download_and_extract(url, save_dir, dataset_filename)
    adata = ad.read_h5ad(file_path)
    return adata

def embedding_small(save_dir='./data'):
    """
    Download, unzip, and load a small version of Embedding dataset as a demo dataset.
    
    Returns
    -------
    pd.DataFrame
        The example dataset loaded into a pandas DataFrame.
    """
    url = 'https://github.com/PancakeZoy/scouter_misc/raw/main/data/Data_Demo/Embedding_small.csv.gz'
    dataset_filename = 'Embedding_small.csv.gz'
    file_path = download_and_extract(url, save_dir, dataset_filename)
    df = pd.read_csv(file_path, index_col=0)
    return df