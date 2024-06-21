from _utils import split_TrainVal, gears_loss
from _model import GenePerturbationModel
from _datasets import BalancedDataset
import anndata as ad
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

class scOracle():
    def __init__(
        self,
        adata,
        embd,
        key_label,
        key_embd_index,
        key_var_genename,
        key_uns,
        device = 'auto'
    ):
        """
        Initialize the model.
        

        Parameters
        ----------
        - adata: 
            Annotated data object.
        - embd: 
            Gene embedding pandas dataframe, where gene names as rownames .
        - key_label:
            The column name of 'adata.obs' that contains the perturbation condition
        -key_embd_index:
            The column name of 'adata.obs' that contains the index of perturbed genes in the gene embedding matrix
        -key_var_genename:
            The column name of 'adata.var' that contains the gene names corresponding to that in gene embedding matrix
        -key_uns:
            The column name of 'adata.obs' that contains the key names of 'adata.uns'
        """
        if not isinstance(adata, ad.AnnData):
            raise TypeError("adata must be an AnnData object")
        if not isinstance(embd, pd.DataFrame):
            raise TypeError("embd must be an pandas DataFrame")
        
        
        self.adata = adata
        self.embd = embd
        self.key_label = key_label
        self.key_embd_index = key_embd_index
        self.key_var_genename = key_var_genename
        self.key_uns = key_uns
        self.n_genes = adata.shape[1]
        self.embd_tensor = torch.tensor(embd.values, dtype=torch.float32)
        self.network = None
        self.nonzero_idx_dict = self.adata.uns['non_zeros_gene_idx']
        self.loss_history = {
            'train_loss': [],
            'val_loss': []
        }
        # Determine the device
        if device == 'auto':
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                self.device = torch.device("cuda:" + str(current_device))
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        
    def train(self, 
              val_conds_include=None,
              val_ratio=0.2,
              batch_size=128,
              n_hidden_encoder=(2048, 512),
              n_out_encoder=64,
              n_hidden_generator=(2048, 3072),
              use_batch_norm=True,
              use_layer_norm=False,
              dropout_rate=0.,
              loss_gamma=2.0,
              loss_lambda=0.1,
              lr=0.005,
              sched_gamma=0.8,
              n_epochs=40,
              patience=5):
        
        train_adata, val_adata = split_TrainVal(self.adata, self.key_label, val_conds_include=val_conds_include, val_ratio=val_ratio)
        train_dataset = BalancedDataset(train_adata, key_label=self.key_label, key_embd_index=self.key_embd_index)
        val_dataset = BalancedDataset(val_adata, key_label=self.key_label, key_embd_index=self.key_embd_index)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True) if len(val_adata) > 0 else None
        
        network = GenePerturbationModel(self.n_genes, 
                                        self.embd_tensor,
                                        n_hidden_encoder=n_hidden_encoder, 
                                        n_out_encoder=n_out_encoder, 
                                        n_hidden_generator=n_hidden_generator,
                                        use_batch_norm=use_batch_norm, 
                                        use_layer_norm=use_layer_norm,
                                        dropout_rate=dropout_rate).to(self.device)
        
        optimizer = optim.Adam(network.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=sched_gamma)
        
        best_val_loss = np.inf
        epochs_no_improve = 0
        best_model_state_dict = None
        
        for epoch in range(n_epochs):
            # Training phase
            network.train()
            train_loss = 0.0
            train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Training Batches", leave=True, unit="batch")
            for ctrl_expr, true_expr, pert_idx, bcode in train_progress:
                ctrl_expr, true_expr, pert_idx = ctrl_expr.to(self.device), true_expr.to(self.device), pert_idx.to(self.device)
                group = train_adata[bcode,:].obs[self.key_uns].values.tolist()
                optimizer.zero_grad()
                pred_expr = network(pert_idx, ctrl_expr)
                loss = gears_loss(pred_expr, true_expr, ctrl_expr, group, 
                                  nonzero_idx_dict=self.nonzero_idx_dict,
                                  gamma=loss_gamma, lambda_=loss_lambda)
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
    
            val_loss = None
            if val_loader:
                # Validation phase
                network.eval()
                val_loss = 0.0
                val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Validation Batches", leave=True, unit="batch")
                with torch.no_grad():
                    for ctrl_expr, true_expr, pert_idx, bcode in val_progress:
                        ctrl_expr, true_expr, pert_idx = ctrl_expr.to(self.device), true_expr.to(self.device), pert_idx.to(self.device)
                        pred_expr = network(pert_idx, ctrl_expr)
                        loss = gears_loss(pred_expr, true_expr, ctrl_expr, group, 
                                          nonzero_idx_dict=self.nonzero_idx_dict,
                                          gamma=loss_gamma, lambda_=loss_lambda)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
    
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
                improvement = best_val_loss - val_loss
                if improvement > 0.001:  # Check if the improvement is greater than 0.001
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_model_state_dict = network.state_dict()
                else:
                    epochs_no_improve += 1
    
                if epochs_no_improve == patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
            
        # Load the best model
        if best_model_state_dict is None:
            print("No improvement observed, keeping the original model")
            self.network = network
        else:
            network.load_state_dict(best_model_state_dict)
            self.network = network


    def pred(self, input_adata):
        # Extract whole prediction datasets
        pred_adata = input_adata.copy()
        pred_dataset = BalancedDataset(pred_adata, key_label=self.key_label, key_embd_index=self.key_embd_index)
        pred_loader = DataLoader(pred_dataset, batch_size=len(pred_dataset), shuffle=False)
        ctrl_expr, true_expr, pert_idx, bcode = next(iter(pred_loader))
        
        # Inference
        self.network.eval()
        with torch.no_grad():
            ctrl_expr, true_expr, pert_idx = ctrl_expr.to(self.device), true_expr.to(self.device), pert_idx.to(self.device)
            prediction = self.network(pert_idx, ctrl_expr).cpu()  # Move prediction back to CPU
        pred_adata = pred_adata[bcode,:]
        pred_adata.X = prediction
        
        # Loss
        # test_loss = gears_loss(prediction, true_expr, ctrl_expr, pert_idx, gamma=loss_gamma, lambda_=loss_lambda).item()

        return pred_adata
    
    def barplot(self, condition, pred_adata, true_adata, key_condition='condition_name', degs_key='top_non_dropout_de_20'):
        ensembl_to_alias = true_adata.var[self.key_var_genename].to_dict()
        alias_to_ensembl = {val:key for key,val in ensembl_to_alias.items()}

        perturbed_alias = np.setdiff1d(condition.split('_')[1].split('+'), 'ctrl')
        perturbed_ensem = [alias_to_ensembl[g] for g in perturbed_alias]
        degs = true_adata.uns[degs_key][condition]
        degs = np.setdiff1d(degs, perturbed_ensem)
        
        g_pred = pred_adata[pred_adata.obs[key_condition]==condition, degs].X.toarray()
        g_true = true_adata[true_adata.obs[key_condition]==condition, degs].X.toarray()
        g_ctrl = true_adata[true_adata.obs[self.key_label]=='ctrl', degs].X.toarray()
        
        # Create a DataFrame for each ndarray
        degs_name = [ensembl_to_alias[degs[i]] for i in range(g_pred.shape[1])]
        df_pred = pd.DataFrame(g_pred, columns=degs_name).assign(Group='Pred')
        df_true = pd.DataFrame(g_true, columns=degs_name).assign(Group='True')
        df_ctrl = pd.DataFrame(g_ctrl, columns=degs_name).assign(Group='Ctrl')
        # Combine all dataframes into one
        df_combined = pd.concat([df_ctrl, df_pred, df_true], axis=0)
        # Melt the dataframe to long format
        df_melted = df_combined.melt(id_vars='Group', var_name='Gene', value_name='Value')
    
        # Create the boxplot
        plt.figure(figsize=(15, 8))
        sns.boxplot(x='Gene', y='Value', hue='Group', data=df_melted, showfliers=False)
        plt.xticks(rotation=90)
        plt.title(f'Expression of top 20 genes after perturbation on {"+".join(perturbed_alias)}')
        plt.show()
        
        
    def evaluate(self,  
                 pred_adata, 
                 true_adata, 
                 key_condition='condition_name', 
                 degs_key='top_non_dropout_de_20'):

        ensembl_to_alias = true_adata.var[self.key_var_genename].to_dict()
        alias_to_ensembl = {val:key for key,val in ensembl_to_alias.items()}
        
        metric = {'NormMSE':{}, 'Pearson':{}, 'Spearman':{}}
        for condition in list(pred_adata.obs[key_condition].unique()):
            perturbed_g = np.setdiff1d(condition.split('_')[1].split('+'), 'ctrl')
            perturbed_g = [alias_to_ensembl[g] for g in perturbed_g]
            degs = true_adata.uns[degs_key][condition]
            degs = np.setdiff1d(degs, perturbed_g)
    
            g_pred = pred_adata[pred_adata.obs[key_condition]==condition, degs].X.toarray().mean(axis=0)
            g_true = true_adata[true_adata.obs[key_condition]==condition, degs].X.toarray().mean(axis=0)
            g_ctrl = true_adata[true_adata.obs[self.key_label]=='ctrl', degs].X.toarray().mean(axis=0)
            
            metric['NormMSE'][condition] = mse(g_true, g_pred)/mse(g_true, g_ctrl)
            metric['Pearson'][condition] = pearsonr(g_true-g_ctrl, g_pred-g_ctrl)[0]
            metric['Spearman'][condition] = spearmanr(g_true, g_pred)[0]
            
        return pd.DataFrame(metric)
            