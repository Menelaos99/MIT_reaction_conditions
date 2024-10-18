import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, Sequential
from torch_geometric.nn import global_mean_pool

def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class NN1(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, num_hidden_layers=None, output_size=None):
        super(NN1, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        
        x = self.output_layer(x)
        return x

class NN(nn.Module):
    def __init__(self, input_dim, n_layers, n_units):
        super(NN, self).__init__()
        self.n_layers = n_layers
        self.n_units = n_units
        
        layers = []
        in_features = input_dim
        for i in range(self.n_layers):
            layers.append(nn.Linear(in_features, n_units[i]))
            layers.append(nn.ReLU())
            in_features = n_units[i]
        layers.append(nn.Linear(in_features, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class GNN(torch.nn.Module):
    def __init__(self, in_channels=None, hidden_channels=None, nn_layers=None, nn_units=None, conv_steps=1, conv_type='GCN'):
        super(GNN, self).__init__()
        self.conv_type = conv_type
        self.gcn = GCNConv(in_channels, hidden_channels)
        self.gat = GATConv(in_channels, hidden_channels)
        self.gat_layers = torch.nn.ModuleList([GATConv(in_channels if i == 0 else hidden_channels, hidden_channels) for i in range(conv_steps)])
        self.bn = torch.nn.BatchNorm1d(hidden_channels)
        # i think we will mostly use the GAT and GCN
        self.gin = GINConv(
            Sequential('x, edge_index', [
                (torch.nn.Linear(in_channels, hidden_channels), 'x -> x'),
                torch.nn.ReLU(),
                (torch.nn.Linear(hidden_channels, hidden_channels), 'x -> x')
            ])
        )
        # self.lin = torch.nn.Linear(hidden_channels, 1)
        self.lin = NN(hidden_channels, n_layers=nn_layers, n_units=nn_units)

    def forward(self, x, edge_index, batch):
        if self.conv_type == 'GCN':
            x = self.gcn(x, edge_index) 
        elif self.conv_type == 'GAT':
            for gat_layer in self.gat_layers:
                x = gat_layer(x, edge_index)
        elif self.conv_type == 'GIN':
            x = self.gin(x, edge_index)
        
        x = self.bn(x)
        x = global_mean_pool(x, batch)

        x = self.lin(x)
        return x
