import pickle
import anndata as ad
import pandas as pd
from _utils import *
from api import model

data_path = '/Users/pancake/Downloads/PerturbGPT/PerturbData/dixit/perturb_processed.h5ad'
embd_path = '/Users/pancake/Downloads/PerturbGPT/PerturbData/GeneEmb/GenePT_emb/GenePT_gene_embedding_ada_text.pickle'

adata = ad.read_h5ad(data_path)
adata.obs['condition'] = adata.obs['condition'].str.replace(r'\+ctrl$', '', regex=True)
with open(embd_path, 'rb') as f:
    embd = pd.DataFrame(pickle.load(f)).T
adata, embd, matched_genes = setup_ad(adata, embd, 'condition', 'gene_name', 'embd_index')


train_adata, test_adata = split_TrainVal(adata, key_label='condition', val_conds_include=None, val_ratio=0.1)

mymodel = model(adata = train_adata, 
                embd = embd, 
                key_label='condition',
                key_embd_index = 'embd_index',
                key_var_genename = 'gene_name')

mymodel.train(val_ratio=0.1, n_epochs=4)

# pred_adata, test_loss = mymodel.pred(test_adata)
pred_adata = mymodel.pred(test_adata)
ensembl_dict = pred_adata.var.to_dict()['gene_name']

list(ensembl_dict.keys())[list(ensembl_dict.values()).index('CHAC1')]

test_adata.obs.condition.unique()
mymodel.barplot('TOR1AIP1', pred_adata=pred_adata, true_adata=test_adata)


pred_adata[pred_adata.obs.condition=='ELK1', ['ENSG00000112306']].X.toarray().mean()
test_adata[test_adata.obs.condition=='ELK1', ['ENSG00000112306']].X.toarray().mean()
train_adata[train_adata.obs.condition=='ctrl', ['ENSG00000112306']].X.toarray().mean()

pred_adata.var[pred_adata.var.gene_name == 'PSMD4']
