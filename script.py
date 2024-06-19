import pickle
import torch
import anndata as ad
import pandas as pd
import numpy as np
from _utils import setup_ad, split_TrainVal
import random
from api import model

# Function to set seeds
def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# Set seeds for reproducibility
set_seeds(42)

data_path = '/Users/pancake/Downloads/PerturbData/adamson/perturb_processed.h5ad'
embd_path = '/Users/pancake/Downloads/PerturbData/GeneEmb/GenePT_emb/GenePT_gene_embedding_ada_text.pickle'

# Load the processed scRNA-seq dataset as Anndata
adata = ad.read_h5ad(data_path)
# Load the gene embedding as the dataframe, and rename its gene alias to match the Anndata
with open(embd_path, 'rb') as f:
    embd = pd.DataFrame(pickle.load(f)).T
ctrl_row = pd.DataFrame([np.zeros(embd.shape[1])], columns=embd.columns, index=['ctrl'])
embd = pd.concat([ctrl_row, embd])
embd.rename(index={'MAP3K21': 'KIAA1804', 'RHOXF2B': 'RHOXF2BB'}, inplace=True)
# Setup anndata to meet the format requirement of model
adata, embd, matched_genes = setup_ad(adata, embd, 'condition', 'gene_name', 'embd_index')
# Split the dataset into train (train+val) and test
all_conds = np.setdiff1d(adata.obs['condition'].unique().tolist(), 'ctrl').tolist()
np.random.shuffle(all_conds)
subsets = np.split(all_conds, range(4, len(all_conds), 4))
subsets

i = 0
metric_list = []
for ss in subsets:
    i += 1
    print(f'{i}/{len(subsets[10:])}')
    train_adata, test_adata = split_TrainVal(adata, key_label='condition', val_conds_include=ss)
    # Initialize the model
    mymodel = model(adata = train_adata, 
                    embd = embd, 
                    key_label='condition',
                    key_embd_index = 'embd_index',
                    key_var_genename = 'gene_name',
                    key_uns = 'condition_name',
                    device='mps')
    # Train the model
    mymodel.train(val_ratio=0., n_epochs=20)
    # Predict on test data
    pred_adata = mymodel.pred(test_adata)
    # Make barplot of a given gene
    # mymodel.barplot('K562(?)_EIF2B3+ctrl_1+1', pred_adata=pred_adata, true_adata=test_adata)
    # Calculate the metric
    metric_df = mymodel.evaluate(pred_adata=pred_adata, true_adata=test_adata)
    metric_list.append(metric_df)
    
metric_df = pd.concat(metric_list)
metric_df.to_csv('Result/metric_adamson_own.csv')
