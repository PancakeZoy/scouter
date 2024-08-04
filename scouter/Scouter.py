from ._utils import gears_loss
from ._model import GenePerturbationModel
from ._datasets import BalancedDataset
from .ScouterData import ScouterData
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr, spearmanr

class Scouter():
    """
    Scouter model class
    
    Attributes
    ----------
    train_adata: anndata.AnnData
        AnnData object for the train split
    val_adata: anndata.AnnData
        AnnData object for the validation split        
    test_adata: anndata.AnnData
        AnnData object for the test split
    embd_tensor: torch.tensor
        torch.tensor object of the gene embedding matrix
    key_label: str
        The column name of `adata.obs` that corresponds to perturbation conditions
    key_var_genename: str
        The column name of `adata.var` that corresponds to gene names.
    key_embd_index: str
        The column name of `adata.obs` that corresponds to gene index in embedding matrix.
    n_genes: int
        Number of genes in the cell expression 
    network: GenePerturbationModel
        The model achieves minimal validation loss after training
    loss_history: dict
        Dictionary containing the loss history on both train split and validation split
    """
    
    def __init__(
        self,
        pertdata: ScouterData,
        device: str='auto'
    ):
        """
        Parameters
        ----------
        - pertdata:
            An ScouterData Object containing cell expression anndata and gene embedding matrix
        - device:
            Device to run the model on. Default: 'auto'
        """

        if not isinstance(pertdata, ScouterData):
            raise TypeError("`pertdata` must be an ScouterData object")
        
        self.embd_idx_dict = {gene:i for i, gene in enumerate(pertdata.embd.index)}
        self.embd_tensor = torch.tensor(pertdata.embd.values, dtype=torch.float32)
        
        self.key_label = pertdata.key_label
        self.key_embd_index = pertdata.key_embd_index
        self.key_var_genename = pertdata.key_var_genename
        self.n_genes = pertdata.train_adata.shape[1]
        
        self.all_adata = pertdata.adata
        self.train_adata = pertdata.train_adata
        self.val_adata = pertdata.val_adata
        self.test_adata = pertdata.test_adata
        self.ctrl_adata = self.train_adata[self.train_adata.obs[self.key_label] == 'ctrl']
        # Determine the device
        if device == 'auto':
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                self.device = torch.device("cuda:" + str(current_device))
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)


    def model_init(self,
                   n_hidden_encoder=(2048, 512),
                   n_out_encoder=64,
                   n_hidden_generator=(2048, 3072),
                   use_batch_norm=True,
                   use_layer_norm=False,
                   dropout_rate=0.):
        self.network = GenePerturbationModel(self.n_genes, 
                                             self.embd_tensor,
                                             n_hidden_encoder=n_hidden_encoder, 
                                             n_out_encoder=n_out_encoder, 
                                             n_hidden_generator=n_hidden_generator,
                                             use_batch_norm=use_batch_norm, 
                                             use_layer_norm=use_layer_norm,
                                             dropout_rate=dropout_rate).to(self.device)
        self.best_val_loss = np.inf
        self.loss_history = {'train_loss': [], 'val_loss': []}


    def train(self,
              nonzero_idx_key='gene_idx_non_zeros',
              batch_size=128,
              loss_gamma=2.0,
              loss_lambda=0.1,
              lr=0.005,
              sched_gamma=0.9,
              n_epochs=40,
              patience=5):
        """
        -nonzero_idx_key:
            The key name of 'adata.uns' that contains the index of non-zero genes in each perturbation group (needed for loss calculation)
        """
        nonzero_idx_dict = self.train_adata.uns[nonzero_idx_key]
        train_dataset = BalancedDataset(self.train_adata, key_label=self.key_label, key_embd_index=self.key_embd_index)
        val_dataset = BalancedDataset(self.val_adata, key_label=self.key_label, key_embd_index=self.key_embd_index)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False) if len(val_dataset) > 0 else None
        
        optimizer = optim.Adam(self.network.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=sched_gamma)
        
        epochs_no_improve = 0
        best_model_state_dict = None
        
        for epoch in range(n_epochs):
            # Training phase
            self.network.train()
            train_loss = 0.0
            train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Training Batches", leave=True, unit="batch")
            for ctrl_expr, true_expr, pert_idx, bcode in train_progress:
                ctrl_expr, true_expr, pert_idx = ctrl_expr.to(self.device), true_expr.to(self.device), pert_idx.to(self.device)
                group = self.train_adata[bcode,:].obs[self.key_label].values.tolist()
                optimizer.zero_grad()
                pred_expr = self.network(pert_idx, ctrl_expr)
                loss = gears_loss(pred_expr, true_expr, ctrl_expr, group, 
                                  nonzero_idx_dict=nonzero_idx_dict,
                                  gamma=loss_gamma, lambda_=loss_lambda)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
    
            if val_loader:
                # Validation phase
                self.network.eval()
                val_loss = 0.0
                val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Validation Batches", leave=True, unit="batch")
                with torch.no_grad():
                    for ctrl_expr, true_expr, pert_idx, bcode in val_progress:
                        ctrl_expr, true_expr, pert_idx = ctrl_expr.to(self.device), true_expr.to(self.device), pert_idx.to(self.device)
                        group = self.val_adata[bcode,:].obs[self.key_label].values.tolist()
                        pred_expr = self.network(pert_idx, ctrl_expr)
                        loss = gears_loss(pred_expr, true_expr, ctrl_expr, group, 
                                          nonzero_idx_dict=nonzero_idx_dict,
                                          gamma=loss_gamma, lambda_=loss_lambda)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
            else:
                val_loss = None
    
            # Store the loss in the history and print
            self.loss_history['train_loss'].append(train_loss)
            if val_loss is not None:
                self.loss_history['val_loss'].append(val_loss)
                print(f'Epoch {epoch+1}/{n_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
            else:
                print(f'Epoch {epoch+1}/{n_epochs}, Training Loss: {train_loss:.4f}')
            
            # Step the learning rate scheduler
            scheduler.step()
    
            # Early stopping logic
            if val_loss is not None:
                improvement = self.best_val_loss - val_loss
                if improvement > 0.001:  # Check if the improvement is greater than 0.001
                    self.best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_model_state_dict = self.network.state_dict()
                else:
                    epochs_no_improve += 1
    
                if epochs_no_improve == patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break

        # Load the best model
        if best_model_state_dict is None:
            print("No improvement observed, keeping the original model")
        else:
            self.network.load_state_dict(best_model_state_dict)


    def pred(self, pert_list, n_pred=300, seed=24):
        np.random.seed(seed)
        # Examine if there is any input gene not in the embedding matrix
        unique_inputs = np.unique(sum([p.split('+') for p in pert_list], []))
        unique_not_embd = [p not in self.embd_idx_dict for p in unique_inputs]
        not_found = unique_inputs[unique_not_embd]
        if len(not_found) > 0:
            raise ValueError(f'{len(not_found)} gene(s) are not found in the gene embedding matrix: {not_found}')
        
        # Prepare the input for network
        all_pairs = pert_list * n_pred
        pert_idx_list = [[self.embd_idx_dict[g] for g in p.split('+')] for p in all_pairs]
        pert_idx = torch.tensor(pert_idx_list, dtype=torch.long).to(self.device)
        ctrl_idx = np.random.choice(range(len(self.ctrl_adata)), size=len(all_pairs), replace=True)
        ctrl_expr = torch.tensor(self.ctrl_adata[ctrl_idx].X.toarray(), dtype=torch.float32).to(self.device)
        
        # Inference
        self.network.eval()
        with torch.no_grad():
            prediction = self.network(pert_idx, ctrl_expr).cpu()  # Move prediction back to CPU
 
        pert_return = {pert:prediction[[i for i, pair in enumerate(all_pairs) if pair == pert]].numpy() for pert in pert_list}
        return pert_return
    
    def barplot(self, condition, degs_key='top20_degs_non_dropout'):
        condition_isin = condition in self.all_adata.obs[self.key_label].unique()
        
        pred = self.pred([condition])[condition]
        ctrl = self.ctrl_adata.X.toarray()
        if not condition_isin:
            print(f'{condition} is not observed.\nTrue prediction expression will be missing. \nTop20 genes are selected based on abs(Prediction-Control)')
            degs = np.argpartition(abs(pred.mean(axis=0) - ctrl.mean(axis=0)), -20)[-20:]
            true = None
            df_true=None
        else:
            degs = self.all_adata.uns[degs_key][condition]
            true = self.all_adata[self.all_adata.obs[self.key_label]==condition].X.toarray()
        degs = np.setdiff1d(degs, np.where(np.isin(self.all_adata.var[self.key_var_genename].values, condition.split('+'))))
        degs_name = self.all_adata.var[self.key_var_genename].values[degs]
        
        df_pred = pd.DataFrame(pred[:, degs], columns=degs_name).assign(Group='Pred')
        df_ctrl = pd.DataFrame(ctrl[:, degs], columns=degs_name).assign(Group='Ctrl')
        df_true = pd.DataFrame(true[:, degs], columns=degs_name).assign(Group='True')
        
        df_combined = pd.concat([df_ctrl, df_pred, df_true], axis=0)
        df_melted = df_combined.melt(id_vars='Group', var_name='Gene', value_name='Value')
        
        # Create the boxplot
        plt.figure(figsize=(15, 8))
        sns.boxplot(x='Gene', y='Value', hue='Group', data=df_melted, showfliers=False)
        plt.xticks(rotation=90)
        plt.title(f'Expression of top 20 genes for perturbation on {condition}')
        plt.show()
        
        
    def evaluate(self,  
                 degs_key='top20_degs_non_dropout'):
        metric = {'NormMSE':{}, 'Pearson':{}, 'Spearman':{}}
        test_conds = list(self.test_adata.obs[self.key_label].unique())
        test_conds.remove('ctrl')
        for condition in test_conds:
            degs = self.all_adata.uns[degs_key][condition]
            degs = np.setdiff1d(degs, np.where(np.isin(self.all_adata.var[self.key_var_genename].values, condition.split('+'))))
        
            pred = self.pred([condition])[condition][:, degs].mean(axis=0)
            ctrl = self.ctrl_adata.X.toarray()[:, degs].mean(axis=0)
            true = self.all_adata[self.all_adata.obs[self.key_label]==condition].X.toarray()[:, degs].mean(axis=0)
            
            metric['NormMSE'][condition] = mse(true, pred)/mse(true, ctrl)
            metric['Pearson'][condition] = pearsonr(true-ctrl, pred-ctrl)[0]
            metric['Spearman'][condition] = spearmanr(true, pred)[0]
        
        return pd.DataFrame(metric)
            