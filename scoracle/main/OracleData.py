import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
from ._utils import split_TrainVal

class OracleData():
    """
    Class for loading and processing perturbation data

    Attributes
    ----------
    adata: anndata.AnnData
        AnnData object containing all cells
    embd: pandas.DataFrame
        pandas dataframe containing gene embeddings
    key_label: str
        The column name of `adata.obs` that corresponds to perturbation conditions
    key_var_genename: str
        The column name of `adata.var` that corresponds to gene names.
    key_embd_index: str
        The column name of `adata.obs` that corresponds to gene index in embedding matrix.
    matched_genes: list
        A list of matched genes between adata and embd
    train_conds: list
        List of perturbation conditions in the train split.
    train_adata: anndata.AnnData
        AnnData object for the train split
    val_conds: list
        List of perturbation conditions in the validation split.
    val_adata: anndata.AnnData
        AnnData object for the validation split        
    test_conds: list
        List of perturbation conditions in the test split.
    test_adata: anndata.AnnData
        AnnData object for the test split
    """    
    def __init__(
        self,
        adata: ad.AnnData,
        embd: pd.DataFrame, 
        key_label: str, 
        key_var_genename: str):
        """ 
        Parameters
        ----------
        adata : AnnData
            Annotated data object. `adata.obs` must contain a column `key_label` with required format: \n
                 `'ctrl'` for control cells \n
                 `'geneA+geneB'` or `'geneA+ctrl'` to denote the name of gene(s) perturbed
        embd : pd.DataFrame
            Gene embedding matrix, with gene names as row names.
        key_label:
            The column name of `adata.obs` that corresponds to perturbation conditions
        key_var_genename:
            The column name of `adata.var` that corresponds to gene names.
        """

        if not isinstance(adata, ad.AnnData):
            raise TypeError("adata must be an AnnData object")
        if not isinstance(embd, pd.DataFrame):
            raise TypeError("embd must be an pandas DataFrame")
        
        self.adata = adata.copy()
        self.embd = embd.copy()
        self.key_label = key_label
        self.key_var_genename = key_var_genename


    def setup_ad(self,
                 key_embd_index: str='embd_index'):

        """
        Setup `adata` and `embd`.
        `embd` will be filtered so that it only contains the matched genes.
        `adata` will drop the perturbation conditions not covered by matched genes.
        A new column `key_embd_index` will be added to `adata.obs`, denoting the index of perturbed genes in `embd`.
        
        Parameters
        ----------
        key_embd_index:
            The column name of `adata.obs` that corresponds to gene index in embedding matrix.
        """
        self.key_embd_index = key_embd_index
        # Find the matched genes between embd and adata, slim embedding matrix to only contain matched genes
        gene_name_ad = self.adata.var[self.key_var_genename].tolist()
        gene_name_embd = self.embd.index.tolist()
        matched_genes = ['ctrl'] + sorted(list(np.intersect1d(gene_name_embd, gene_name_ad)))
        self.embd = self.embd.loc[matched_genes]
        print(f'{len(matched_genes)} matched genes found between your dataset and gene embedding')

        # Detect any perturbed genes that are not in matched genes
        uniq_conds = self.adata.obs[self.key_label].unique().tolist()
        if 'ctrl' not in uniq_conds:
            raise TypeError("Provided annData does not have control cells")
        perturb_genes = np.unique(sum([p.split('+') for p in uniq_conds], []))
        unmatched_genes = perturb_genes[[p not in matched_genes for p in perturb_genes]]
        if len(unmatched_genes) > 0:
            print(f'{len(unmatched_genes)} perturbed genes are not found in the gene embedding matrix: {unmatched_genes}. \nHence they are deleted.\nPlease check if this is because of different gene synonyms. ')
            # Filter the DataFrame by excluding rows where the condition contains unmatched genes
            self.adata = self.adata[~self.adata.obs[self.key_label].str.contains('|'.join(unmatched_genes))]
            # self.adata = self.adata[~self.adata.obs[self.key_label].apply(lambda condition: any(gene in condition for gene in unmatched_genes))].copy()            
            uniq_conds = self.adata.obs[self.key_label].unique().tolist()
            perturb_genes = np.unique(sum([p.split('+') for p in uniq_conds], []))
        else:
            print('All perturbed genes are found in the gene embedding matrix!')
            
        #Create a new column that contains the index of perturbed genes in embd matrix
        gene_ind_dic = {g: matched_genes.index(g) for g in perturb_genes}
        cond_ind_dic = {cond:[gene_ind_dic[gene] for gene in cond.split('+')] for cond in uniq_conds}
        self.adata.obs[key_embd_index] = self.adata.obs[self.key_label].astype(str).map(cond_ind_dic)
        self.matched_genes = matched_genes

    def split_Train_Val_Test(
            self, 
            val_ratio=0.1,
            if_test = True,
            test_conds=None,
            test_ratio=0.2, 
            seed=24):
        """
        Splits the annotated data into training, validation and testing sets.
        
        Parameters
        ----------
        val_ratio: float
            The proportion of the validation split compared to train split. Default is 0.1.
        if_test: boolean
            Wheter to generate a split for testing. Default is True.
        test_conds : list or None, optional
            List of perturbation conditions to be the test set. If None, conditions are selected randomly based on `test_ratio`. Default is None.
        test_ratio : float, optional
            The proportion of the test split compared to the rest. Default is 0.2.
        seed : int, optional
            Random seed for reproducibility. Default is 24.
        """
        self.train_conds, self.train_adata, self.val_conds, self.val_adata, self.test_conds, self.test_adata = \
            None, None, None, None, None, None
        if if_test:
            _, train_val_adata, self.test_conds, self.test_adata = \
                split_TrainVal(self.adata, self.key_label, val_conds_include=test_conds, val_ratio=test_ratio, seed=seed)
            self.train_conds, self.train_adata, self.val_conds, self.val_adata = \
                split_TrainVal(train_val_adata, self.key_label, val_conds_include=None, val_ratio=val_ratio, seed=seed)
        else:
            self.train_conds, self.train_adata, self.val_conds, self.val_adata = \
                split_TrainVal(self.adata, self.key_label, val_conds_include=None, val_ratio=val_ratio, seed=seed)

    def gene_ranks(self, rankby_abs=True, **kwargs):
        """
        Rank genes for each perturbation group. Saved as a dictionary in `adata.uns['rank_genes_groups']`

        Parameters
        ----------
        rankby_abs: boolean, optional
            Rank genes by the absolute value of the score, not by the score. 
            The returned scores are never the absolute values.
            Default is True.
        kwargs
            All additional keyword arguments passed to the `scanpy.tl.rank_genes_groups` call.
        """
        
        gene_dict = {}
        sc.tl.rank_genes_groups(
            self.adata,
            groupby=self.key_label,
            reference='ctrl',
            rankby_abs=rankby_abs,
            n_genes=len(self.adata.var),
            **kwargs)
        de_genes = pd.DataFrame(self.adata.uns['rank_genes_groups']['names'])
        for group in de_genes:
            gene_dict[group] = de_genes[group].tolist()
        self.adata.uns['rank_genes_groups'] = gene_dict

    def get_dropout_non_zero_genes(self):
        """
        Compute the non-zero genes and top 20 differentially expressed genes for each perturbation group. 
        Saved as dictionaries in\n
        `adata.uns['top20_degs']`\n
        `adata.uns['top20_degs_non_zero']`\n
        `adata.uns['top20_degs_non_dropout']`\n
        `adata.uns['gene_idx_non_dropout']`\n
        `adata.uns['gene_idx_non_zeros']`

        Parameters
        ----------
        rankby_abs: boolean, optional
            Rank genes by the absolute value of the score, not by the score. 
            The returned scores are never the absolute values.
            Default is True.
        kwargs
            All additional keyword arguments passed to the `scanpy.tl.rank_genes_groups` call.
        """
        
        if 'rank_genes_groups' not in self.adata.uns.keys():
            raise ValueError("Gene expression (ad.AnnData) does not have 'rank_genes_groups' in .uns, please first run function gene_ranks()")
        
        ctrl = np.mean(self.adata[self.adata.obs[self.key_label] == 'ctrl'].X, axis = 0)
        gene_id2idx = dict(zip(self.adata.var.index.values, range(len(self.adata.var))))
    
        non_zeros_gene_idx = {}
        non_dropout_gene_idx = {}
        top_non_dropout_de_20 = {}
        top_non_zero_de_20 = {}
        top_de_20 = {}
    
        for pert in self.adata.uns['rank_genes_groups'].keys():
            X = np.mean(self.adata[self.adata.obs[self.key_label] == pert].X, axis = 0)
    
            non_zero = np.where(np.array(X)[0] != 0)[0]
            zero = np.where(np.array(X)[0] == 0)[0]
            true_zeros = np.intersect1d(zero, np.where(np.array(ctrl)[0] == 0)[0])
            non_dropouts = np.concatenate((non_zero, true_zeros))
    
            rank_genes = self.adata.uns['rank_genes_groups'][pert]
            gene_idx_top = [gene_id2idx[g] for g in rank_genes]
            
            de_20 = gene_idx_top[:20]
            non_dropout_20 = [i for i in gene_idx_top if i in non_dropouts][:20]
            non_zero_20 = [i for i in gene_idx_top if i in non_zero][:20]
            de_20 = gene_idx_top[:20]
            
            non_zeros_gene_idx[pert] = non_zero
            non_dropout_gene_idx[pert] = non_dropouts
            top_non_dropout_de_20[pert] = np.array(non_dropout_20)
            top_non_zero_de_20[pert] = np.array(non_zero_20)
            top_de_20[pert] = np.array(de_20)
    
        self.adata.uns['top20_degs'] = top_de_20
        self.adata.uns['top20_degs_non_zero'] = top_non_zero_de_20
        self.adata.uns['top20_degs_non_dropout'] = top_non_dropout_de_20
        self.adata.uns['gene_idx_non_dropout'] = non_dropout_gene_idx
        self.adata.uns['gene_idx_non_zeros'] = non_zeros_gene_idx
