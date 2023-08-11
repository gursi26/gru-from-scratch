import torch
from torch import nn 
import torch.nn.functional as F

class GRUCell(nn.Module):

    def __init__(self, input_dim, hidden_dim) -> None:
        super(GRUCell, self).__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.relevance_whh, self.relevance_wxh, self.relevance_b = self.create_gate_parameters()
        self.update_whh, self.update_wxh, self.update_b = self.create_gate_parameters()
        self.candidate_whh, self.candidate_wxh, self.candidate_b = self.create_gate_parameters()

    def create_gate_parameters(self):
        input_weights = nn.Parameter(torch.zeros(self.input_dim, self.hidden_dim))
        hidden_weights = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim))
        nn.init.xavier_uniform_(input_weights)
        nn.init.xavier_uniform_(hidden_weights)
        bias = nn.Parameter(torch.zeros(self.hidden_dim))
        return hidden_weights, input_weights, bias
    
    def forward(self, x, h):
        output_hiddens = []
        for i in range(x.shape[1]):
            relevance_gate = F.sigmoid((h @ self.relevance_whh) + (x[:, i] @ self.relevance_wxh) + self.relevance_b)
            update_gate = F.sigmoid((h @ self.update_whh) + (x[:, i] @ self.update_wxh) + self.update_b)
            candidate_hidden = F.tanh(((relevance_gate * h) @ self.candidate_whh) + (x[:, i] @ self.candidate_wxh) + self.candidate_b)
            h = (update_gate * candidate_hidden) + ((1 - update_gate) * h)
            output_hiddens.append(h.unsqueeze(1))
        return torch.concat(output_hiddens, dim=1)


class MultiLayerGRU(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(MultiLayerGRU, self).__init__()
        self.input_dim, self.hidden_dim, self.num_layers = input_dim, hidden_dim, num_layers
        self.layers = nn.ModuleList()
        self.layers.append(GRUCell(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GRUCell(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, input_dim)
        nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)

    def forward(self, x, h):
        output_hidden = self.layers[0](x, h[0])
        new_hidden = [output_hidden[:, -1].unsqueeze(0)]
        for i in range(1, self.num_layers):
            output_hidden = self.layers[i](self.dropout(output_hidden), h[i])
            new_hidden.append(output_hidden[:, -1].unsqueeze(0))
        return self.linear(self.dropout(output_hidden)), torch.concat(new_hidden, dim=0)