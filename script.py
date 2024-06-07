import pickle
import anndata as ad
import pandas as pd
from _utils import *
from api import model

adata = ad.read_h5ad('/Users/pancake/Downloads/PerturbGPT/PerturbData/adamson/perturb_processed.h5ad')
adata.obs['condition'] = adata.obs['condition'].str.replace(r'\+ctrl$', '', regex=True)
embd_path = '/Users/pancake/Downloads/PerturbGPT/PerturbData/GeneEmb/GenePT_emb/GenePT_gene_embedding_ada_text.pickle'
with open(embd_path, 'rb') as f:
    embd = pd.DataFrame(pickle.load(f)).T
adata, embd, matched_genes = setup_ad(adata, embd, 'condition', 'gene_name', 'embd_index')


train_adata, test_adata = split_TrainVal(adata, key_label='condition', val_conds_include=None, val_ratio=0.1)

mymodel = model(adata = train_adata, 
                embd = embd, 
                key_label='condition',
                key_embd_index = 'embd_index',
                key_var_genename = 'gene_name')

mymodel.train(val_ratio=0.1, n_epochs=2)

pred_adata, test_loss = mymodel.pred(test_adata)

test_adata.obs.condition.unique()
mymodel.barplot('SEC63', pred_adata=pred_adata, true_adata=test_adata)


pred_adata[pred_adata.obs.condition=='PSMD4', ['ENSG00000159352']].X.toarray().mean()
train_adata[train_adata.obs.condition=='PSMD4', ['ENSG00000159352']].X.toarray().mean()
train_adata[train_adata.obs.condition=='ctrl', ['ENSG00000159352']].X.toarray().mean()

pred_adata.var[pred_adata.var.gene_name == 'PSMD4']
