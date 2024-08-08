import torch
import torch.nn as nn

class ScouterModel(nn.Module):
    def __init__(self, 
                 n_genes: int, 
                 embd, 
                 n_encoder: tuple, 
                 n_out_encoder: int, 
                 n_decoder: tuple,
                 use_batch_norm: bool, 
                 use_layer_norm: bool,
                 dropout_rate: float):
        
        """
        Initialize the ScouterModel.

        Parameters:
        ----------
        - n_genes: 
            Number of input genes.
        - embd: 
            Gene embedding matrix.
        - n_encoder: 
            Tuple specifying the hidden layer sizes for the cell encoder.
        - n_out_encoder: 
            Size of the output layer for the cell encoder.
        - n_decoder: 
            Tuple specifying the hidden layer sizes for the generator.
        - use_batch_norm: 
            Whether to use batch normalization.
        - use_layer_norm: 
            Whether to use layer normalization.
        - dropout_rate: 
            Dropout rate.        
        """
        
        super(ScouterModel, self).__init__()
        self.embd = nn.Parameter(embd.clone().detach(), requires_grad=False)
        n_embd = self.embd.shape[1]
        self.encoder = self._build_mlp(n_genes, 
                                       n_encoder, 
                                       n_out_encoder, 
                                       use_batch_norm, 
                                       use_layer_norm, 
                                       dropout_rate)
        self.generator = self._build_mlp(n_out_encoder + n_embd, 
                                         n_decoder, 
                                         n_genes,
                                         use_batch_norm, 
                                         use_layer_norm, 
                                         dropout_rate)

    def forward(self, pert_idx, ctrl_exp):
        """
        Forward pass of the ScouterModel.
    
        Parameters:
        - pert_idx (torch.Tensor): 
            Tensor containing the indices of perturbed genes. Shape: (batch_size, num_genes).
        - ctrl_exp (torch.Tensor): 
            Tensor containing the control expression values. Shape: (batch_size, num_genes).
        """        
        input_gene = self.embd[pert_idx].sum(axis=1)
        input_ctrl = self.encoder(ctrl_exp)
        concatenated_input = torch.cat((input_gene, input_ctrl), dim=-1)
        output = self.generator(concatenated_input)
        return output
        
    def _build_mlp(self, 
                   input_dim, 
                   hidden_dims, 
                   output_dim, 
                   use_batch_norm, 
                   use_layer_norm, 
                   dropout_rate):
        """
        Builds a multi-layer perceptron (MLP) with the specified configurations.
    
        Parameters:
        - input_dim (int): 
            The dimension of the input layer.
        - hidden_dims (tuple of int): 
            A tuple specifying the size of each hidden layer.
        - output_dim (int): 
            The dimension of the output layer.
        - use_batch_norm (bool): 
            Whether to use batch normalization after each hidden layer.
        - use_layer_norm (bool): 
            Whether to use layer normalization after each hidden layer.
        - dropout_rate (float): 
            The dropout rate to use in Alpha Dropout layers. Set to 0.0 to disable dropout.
    
        Returns:
        - nn.Sequential: 
            A sequential container with the constructed MLP layers.
    
        Example:
        mlp = model._build_mlp(input_dim=128, hidden_dims=(64, 32), output_dim=10, 
                               use_batch_norm=True, use_layer_norm=False, dropout_rate=0.1)
        """        
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            if use_batch_norm:
                # layers.append(nn.BatchNorm1d(h_dim, momentum=0.01, eps=0.001))
                layers.append(nn.BatchNorm1d(h_dim))
            if use_layer_norm:
                # layers.append(nn.LayerNorm(h_dim, elementwise_affine=False))
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.SELU())
            if dropout_rate > 0.:
                layers.append(nn.AlphaDropout(dropout_rate))
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)