import torch
import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc

def gears_loss(predicted, true, control, group, gamma, lambda_):
    def group_mean(expr, group):
        unique_ids, inverse_indices = torch.unique(group, return_inverse=True, dim=0)
        num_groups = unique_ids.size(0)
        group_sums = torch.zeros(num_groups)
        group_sums.index_add_(0, inverse_indices, expr)
        group_counts = torch.zeros(num_groups).index_add_(0, inverse_indices, torch.ones(group.shape[0]))
        group_means = group_sums / group_counts
        return group_means

    # Autofocus loss calculation
    DiffExpr_CellAvg = ((true - predicted).abs() ** (2 + gamma)).mean(axis=1)
    autofocus_loss = group_mean(DiffExpr_CellAvg, group).mean()
    
    # Direction-aware loss calculation    
    DiffSign_CellAvg = ((torch.sign(true - control) - torch.sign(predicted - control))**2).mean(axis=1)
    direction_aware_loss = group_mean(DiffSign_CellAvg, group).mean()

    # Total loss
    total_loss = autofocus_loss + lambda_ * direction_aware_loss
    return total_loss


def split_TrainValTest(adata, key_label, test_conds_include=None, val_test_ratio=[0.1,0.1]):
    train_adata, val_adata = split_TrainVal(adata, key_label, val_conds_include=test_conds_include, val_ratio=sum(val_test_ratio))
    test_by_val = val_test_ratio[1]/sum(val_test_ratio)
    val_adata, test_adata = split_TrainVal(val_adata, key_label, val_conds_include=test_conds_include, val_ratio=test_by_val)
    
    return train_adata, val_adata, test_adata


# def split_TrainVal(adata, key_label, val_conds_include=None, val_ratio=0.2):
#     """Splits the data into train, validation based on conditions."""
#     all_conds = adata[adata.obs[key_label] != 'ctrl'].obs[key_label].unique().tolist()
#     np.random.shuffle(all_conds)
#     n_ValNeed = round(val_ratio * len(all_conds))
#     if val_conds_include is None:
#         val_conds, train_conds = np.split(all_conds, [n_ValNeed])
#         train_conds = list(train_conds)+['ctrl']
#         val_conds = list(val_conds)+['ctrl']
#     else:
#         n_given = len(val_conds_include)
#         if n_given > n_ValNeed:
#             raise ValueError(f"Number of given validation conditions must be smaller than or equal to {n_ValNeed}.")
#         rest_of_conds = list(set(all_conds) - set(val_conds_include))
#         val_conds, train_conds = np.split(rest_of_conds, [(n_ValNeed-n_given)])
#         train_conds = list(train_conds)+['ctrl']
#         val_conds = list(val_conds)+val_conds_include+['ctrl']

#     train_mask = adata.obs['condition'].isin(train_conds)
#     val_mask = adata.obs['condition'].isin(val_conds)
#     train_adata = adata[train_mask]
#     val_adata = adata[val_mask]

#     return train_adata, val_adata

def split_TrainVal(adata, key_label, val_conds_include=None, val_ratio=0.2, seed=42):
    """Splits the data into train, validation based on conditions."""
    all_conds = adata[adata.obs[key_label] != 'ctrl'].obs[key_label].unique().tolist()
    if val_conds_include is None:
        np.random.seed(42)
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

    return train_adata, val_adata



def setup_ad(adata: ad.AnnData, 
             embd: pd.DataFrame, 
             key_label: str, 
             key_Gname: str, 
             key_embd_index: str):
    """
    Setup the Annotated data object and gene embedding matrix. 
    The gene embedding matrix will be filtered to only contain the genes appearing in the adata and ordered by the matched gene names.
    A new column 'key_embd_index' will be added to adata.obs, denoting the index of the perturbed gene in the gene embedding matrix.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object. adata.obs must contain a column 'key_label' with required format: 
             'ctrl' for control cells, or
             gene names to denote the gene perturbed
    embd : pd.DataFrame
        Gene embedding pandas DataFrame, with gene names as row names.
    key_label : str
        Column name in adata.obs that denotes perturbed gene names.
    key_Gname : str
        Column name in adata.var that contains gene names.
    key_embd_index : str
        New column name to be added to adata.obs that will store the index of the perturbed gene in the gene embedding matrix.
    """
    adata = adata.copy(); embd=embd.copy()
    
    # Normalize the condition name. Make "A+B" and "B+A" the same
    adata.obs[key_label] = adata.obs[key_label].astype(str).apply(lambda x: '+'.join(sorted(x.split('+'))))
    adata.obs[key_label] = adata.obs[key_label].astype('category')
    # Find the matched genes between embd and adata
    gene_name_ad = adata.var[key_Gname].tolist()
    gene_name_embd = embd.index.tolist()
    matched_genes = ['ctrl'] + sorted(list(set(gene_name_embd) & set(gene_name_ad)))
    print(f'{len(matched_genes)} matched genes found between your dataset and gene embedding')
    embd_mtx = embd.loc[matched_genes]
    # Detect any perturbed genes that are not in matched genes
    uniq_conds = adata.obs[key_label].unique().tolist()
    if 'ctrl' not in uniq_conds:
        raise TypeError("Provided annData does not have control cells")
    perturb_genes = np.unique(sum([p.split('+') for p in uniq_conds], []))
    unmatched_genes = perturb_genes[[p not in matched_genes for p in perturb_genes]]
    if len(unmatched_genes) > 0:
        print(f'{len(unmatched_genes)} perturbed genes are not found in the gene embedding matrix: {unmatched_genes}. \nHence they are deleted.')
        # Filter the DataFrame to exclude rows where the condition contains any unmatched genes
        adata = adata[~adata.obs[key_label].apply(lambda condition: any(gene in condition for gene in unmatched_genes))].copy()
        uniq_conds = adata.obs[key_label].unique().tolist()
        perturb_genes = np.unique(sum([p.split('+') for p in uniq_conds], []))
    else:
        print('All perturbed genes are found in the gene embedding matrix!')
    #Create a new column that contains the index of perturbed genes in embd matrix
    gene_ind_dic = {g: matched_genes.index(g) for g in perturb_genes}
    cond_ind_dic = {cond:[gene_ind_dic[gene] for gene in cond.split('+')] for cond in uniq_conds}
    adata.obs[key_embd_index] = adata.obs[key_label].astype(str).map(cond_ind_dic)
    
    return adata, embd_mtx, matched_genes



def TopKGens(adata, fashion= 'symm',
             n_genes = 20, 
             groupby='condition', 
             reference='ctrl', 
             method = 'wilcoxon'):
    if fashion == 'symm':
        sc.tl.rank_genes_groups(adata,  
                                groupby=groupby,
                                reference = reference,
                                method = method)
        df = sc.get.rank_genes_groups_df(adata, None)
        symm_K = {}
        for g in df.group.unique().tolist():
            g_df = df[df['group']==g]
            degs = g_df.nlargest(10, 'scores')['names'].values.tolist() + \
                g_df.nsmallest(10, 'scores')['names'].values.tolist()
            symm_K[g] = degs
        adata.uns['symm_top_degs'] = symm_K
    else:
        sc.tl.rank_genes_groups(adata, 
                                n_genes = n_genes, 
                                groupby=groupby,
                                reference = reference,
                                method = method,
                                rankby_abs = True)
        df = sc.get.rank_genes_groups_df(adata, None)
        abs_K = {}
        for g in df.group.unique().tolist():
            degs = df[df['group']==g].names.values.tolist()
            abs_K[g] = degs
        adata.uns['abs_top_degs'] = abs_K
    del adata.uns['rank_genes_groups']
