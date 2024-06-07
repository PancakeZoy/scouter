from _utils import *
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

class model():
    def __init__(
        self,
        adata,
        embd,
        key_label,
        key_embd_index,
        key_var_genename
    ):
        """
        Initialize the model.
        

        Parameters
        ----------
        - adata: 
            Annotated data object.
        - embd: 
            Gene embedding pandas dataframe, where gene names as rownames .
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
        self.n_genes = adata.shape[1]
        self.embd_tensor = torch.tensor(embd.values, dtype=torch.float32)
        self.network = None
        self.loss_history = {
            'train_loss': [],
            'val_loss': []
        }
        
        
    def train(self, 
              val_conds_include=None,
              val_ratio=0.2,
              batch_size = 128,
              n_hidden_encoder =(2048, 512),
              n_out_encoder =64,
              n_hidden_generator =(2048, 3072),
              use_batch_norm =True,
              use_layer_norm =False,
              dropout_rate = 0.,
              loss_gamma=2.0, 
              loss_lambda_=0.1,
              lr = 0.005,
              sched_gamma = 0.8,
              n_epochs = 40,
              patience = 5):
        
        train_adata, val_adata = split_TrainVal(self.adata, self.key_label, val_conds_include=val_conds_include, val_ratio=val_ratio)
        train_dataset = BalancedDataset(train_adata, key_label=self.key_label, key_embd_index=self.key_embd_index)
        val_dataset = BalancedDataset(val_adata, key_label=self.key_label, key_embd_index=self.key_embd_index)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        network = GenePerturbationModel(self.n_genes, 
                                        self.embd_tensor,
                                        n_hidden_encoder = n_hidden_encoder, 
                                        n_out_encoder = n_out_encoder, 
                                        n_hidden_generator = n_hidden_generator,
                                        use_batch_norm = use_batch_norm, 
                                        use_layer_norm = use_layer_norm,
                                        dropout_rate = dropout_rate)
        
        
        optimizer = optim.Adam(network.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=sched_gamma)
        
        # Start training with early stopping based on validation sets
        best_val_loss = np.inf
        epochs_no_improve = 0
        
        for epoch in range(n_epochs):
            # Training phase
            network.train()
            train_loss = 0.0
            train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Training Batches", leave=True, unit="batch")
            for ctrl_exp, pert_expr, gene_idx, bcode in train_progress:
                optimizer.zero_grad()
                output = network(gene_idx, ctrl_exp)
                loss = gears_loss(output, pert_expr, ctrl_exp, gene_idx, gamma=loss_gamma, lambda_=loss_lambda_)
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validation phase
            network.eval()
            val_loss = 0.0
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Validation Batches", leave=True, unit="batch")
            with torch.no_grad():
                for ctrl_exp, pert_expr, gene_idx, bcode in val_progress:
                    output = network(gene_idx, ctrl_exp)
                    loss = gears_loss(output, pert_expr, ctrl_exp, gene_idx, gamma=loss_gamma, lambda_=loss_lambda_)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
    
            # Store the loss in the history and print
            self.loss_history['train_loss'].append(train_loss)
            self.loss_history['val_loss'].append(val_loss)
            print(f'Epoch {epoch+1}/{n_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
            # Step the learning rate scheduler
            scheduler.step()
        
            # Early stopping logic
            if val_loss < best_val_loss:
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
        else:
            network.load_state_dict(best_model_state_dict)
            self.model = network

    def pred(self, input_adata):
        # Extract whole prediction datasets
        pred_adata = input_adata.copy()
        pred_dataset = BalancedDataset(pred_adata, key_label=self.key_label, key_embd_index=self.key_embd_index)
        pred_loader = DataLoader(pred_dataset, batch_size=len(pred_dataset), shuffle=False)
        ctrl_exp, pert_expr, gene_idx, bcode = next(iter(pred_loader))
        
        # Inference
        prediction = self.model(gene_idx, ctrl_exp).detach()
        pred_adata = pred_adata[bcode,:]
        pred_adata.X = prediction
        
        # Loss
        # test_loss = gears_loss(prediction, pert_expr, ctrl_exp, gene_idx, gamma=loss_gamma, lambda_=loss_lambda_).item()

        # return pred_adata, test_loss
        return pred_adata
    
    def barplot(self, gene_name, pred_adata, true_adata):
        def process_key(key):
            key = key.split('_', 1)[-1]
            key = key.split('+ctrl')[0]
            return key
        topGene_dic = true_adata.uns['top_non_dropout_de_20']
        processed_dic = {process_key(key): value for key, value in topGene_dic.items()}
        degs = processed_dic[gene_name]
        degs_namedict = (true_adata.var.to_dict())[self.key_var_genename]
        
        g_pred = pred_adata[pred_adata.obs[self.key_label]==gene_name, degs].X.toarray()
        g_true = true_adata[true_adata.obs[self.key_label]==gene_name, degs].X.toarray()
        g_ctrl = true_adata[true_adata.obs[self.key_label]=='ctrl', degs].X.toarray()
        
        # Create a DataFrame for each ndarray
        degs_name = [f'{degs_namedict[degs[i]]}' for i in range(g_pred.shape[1])]
        df_pred = pd.DataFrame(g_pred, columns=degs_name)
        df_true = pd.DataFrame(g_true, columns=degs_name)
        df_ctrl = pd.DataFrame(g_ctrl, columns=degs_name)
        # Add a column to identify the source of the data
        df_pred['Group'] = 'Pred'
        df_true['Group'] = 'True'
        df_ctrl['Group'] = 'Ctrl'
        # Combine all dataframes into one
        df_combined = pd.concat([df_ctrl, df_pred, df_true], axis=0)
        # Melt the dataframe to long format
        df_melted = df_combined.melt(id_vars='Group', var_name='Gene', value_name='Value')

        # Create the boxplot
        plt.figure(figsize=(15, 8))
        sns.boxplot(x='Gene', y='Value', hue='Group', data=df_melted, showfliers=False)
        plt.xticks(rotation=90)
        plt.title(f'Expression of top 20 genes after perturbation on {gene_name}')
        plt.show()
        
        
        
        
        
        
        
        