import torch
from torch import nn
import torch.nn.functional as F

def _make_hidden_layer(in_dim, out_dim, activation, dropout=None):
    if dropout:
        return nn.Sequential(nn.Linear(in_dim, out_dim), activation, nn.Dropout(p=dropout))
    return nn.Sequential(nn.Linear(in_dim, out_dim), activation)


class MortgageNetwork(nn.Module):
    """Mortgage Delinquency DNN."""

    def __init__(
        self,
        num_features,
        embedding_size,
        hidden_dims,
        use_cuda=True,
        activation=nn.ReLU(),
        dropout=None,
        embedding_bag_mode='mean'
    ):
        super(MortgageNetwork, self).__init__()
        self.input_size = num_features
        self.embedding_size = embedding_size
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.activation = activation
        self.dropout = dropout

        self.embedding = nn.modules.EmbeddingBag(self.input_size, self.embedding_size,
                                                 mode=embedding_bag_mode)

        if len(hidden_dims) > 0:
            dims = [self.embedding_size] + hidden_dims
            hidden_layers = [
                _make_hidden_layer(dims[i], dims[i + 1], self.activation, self.dropout)
                for i in range(len(dims) - 1)
            ]
            self.hidden_layers = nn.ModuleList(hidden_layers)
            self.hidden_layers.extend([nn.Linear(dims[-1], 1)])
        else:
            self.hidden_layers = []

        self.to(self.device)

    def forward(self, x):
        """Forward pass."""
        out = self.embedding(x)
        out = self.activation(out)
        for layer in self.hidden_layers:
            out = layer(out)
        return out.squeeze()