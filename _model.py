import torch
import torch.nn as nn

class GenePerturbationModel(nn.Module):
    def __init__(self, 
                 n_genes: int, 
                 embd, 
                 n_hidden_encoder: tuple=(2048, 512), 
                 n_out_encoder: int=64, 
                 n_hidden_generator: tuple=(2048, 3072),
                 use_batch_norm: bool=True, 
                 use_layer_norm: bool=False,
                 dropout_rate: float = 0.):
        
        """
        Initialize the GenePerturbationModel.

        Parameters:
        - n_genes: 
            Number of input genes.
        - embd: 
            Gene embedding matrix.
        - n_hidden_encoder: 
            Tuple specifying the hidden layer sizes for the cell encoder.
        - n_out_encoder: 
            Size of the output layer for the cell encoder.
        - n_hidden_generator: 
            Tuple specifying the hidden layer sizes for the generator.
        - use_batch_norm: 
            Whether to use batch normalization.
        - use_layer_norm: 
            Whether to use layer normalization.
        - dropout_rate: 
            Dropout rate.
        """
        
        super(GenePerturbationModel, self).__init__()
        self.embd = nn.Parameter(embd.clone().detach(), requires_grad=False)
        n_embd = self.embd.shape[1]
        self.encoder = self._build_mlp(n_genes, 
                                       n_hidden_encoder, 
                                       n_out_encoder, 
                                       use_batch_norm, 
                                       use_layer_norm, 
                                       dropout_rate)
        self.generator = self._build_mlp(n_out_encoder + n_embd, 
                                         n_hidden_generator, 
                                         n_genes,
                                         use_batch_norm, 
                                         use_layer_norm, 
                                         dropout_rate)

    def forward(self, gene_idx, ctrl_exp):
        input_gene = self.embd[gene_idx].sum(axis=1)
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