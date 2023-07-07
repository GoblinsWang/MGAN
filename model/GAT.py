import math
import torch
from torch import nn
import torch.nn.functional as F
import copy

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_graph):
        nodes_q = self.query(input_graph)
        nodes_k = self.key(input_graph)
        nodes_v = self.value(input_graph)

        nodes_q_t = self.transpose_for_scores(nodes_q)
        nodes_k_t = self.transpose_for_scores(nodes_k)
        nodes_v_t = self.transpose_for_scores(nodes_v)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(nodes_q_t, nodes_k_t.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in GATModel forward() function)
        attention_scores = attention_scores 

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        nodes_new = torch.matmul(attention_probs, nodes_v_t)
        nodes_new = nodes_new.permute(0, 2, 1, 3).contiguous()
        new_nodes_shape = nodes_new.size()[:-2] + (self.all_head_size,)
        nodes_new = nodes_new.view(*new_nodes_shape)
        return nodes_new


class GATLayer(nn.Module):
    def __init__(self, config):
        super(GATLayer, self).__init__()
        self.mha = MultiHeadAttention(config)

        self.fc_in = nn.Linear(config.hidden_size, config.hidden_size)
        self.bn_in = nn.BatchNorm1d(config.hidden_size)
        self.dropout_in = nn.Dropout(config.hidden_dropout_prob)

        self.fc_int = nn.Linear(config.hidden_size, config.hidden_size)
        self.relu = nn.GELU()
        self.fc_out = nn.Linear(config.hidden_size, config.hidden_size)
        self.bn_out = nn.BatchNorm1d(config.hidden_size)
        self.dropout_out = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_graph):
        attention_output = self.mha(input_graph) # multi-head attention
        attention_output = self.fc_in(attention_output)
        attention_output = self.dropout_in(attention_output)
        attention_output = self.bn_in((attention_output + input_graph).permute(0, 2, 1)).permute(0, 2, 1)
        intermediate_output = self.fc_int(attention_output)
        # intermediate_output = F.relu(intermediate_output)
        intermediate_output = self.relu(intermediate_output)
        intermediate_output = self.fc_out(intermediate_output)
        intermediate_output = self.dropout_out(intermediate_output)
        graph_output = self.bn_out((intermediate_output + attention_output).permute(0, 2, 1)).permute(0, 2, 1)
        return graph_output



class GATopt(object):
    def __init__(self, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = 1
        self.hidden_dropout_prob = 0.2
        self.attention_probs_dropout_prob = 0.2


class GAT(nn.Module):
    def __init__(self, config_gat):
        super(GAT, self).__init__()
        layer = GATLayer(config_gat)
        self.encoder = nn.ModuleList([copy.deepcopy(layer) for _ in range(config_gat.num_layers)])
        self.weight1 = nn.Parameter(torch.randn(1))
        # self.weight2 = nn.Parameter(torch.randn(1))
        self.relu = nn.GELU()

    def forward(self, input_graph):
        B, C, H, W = input_graph.shape
        
        # [B H*W C]
        hidden_states = input_graph.reshape(B, C, -1, 1).squeeze(3).transpose(2,1).contiguous()
        # batch = input_graph.shape[0]

        for layer_module in self.encoder:
            hidden_states = layer_module(hidden_states)

        # [B, C, H, W]
        hidden_states = hidden_states.transpose(2,1).unsqueeze(3).reshape(B, C, H, W).contiguous()

        output_graph = self.relu(self.weight1 * hidden_states + (1 - self.weight1) * input_graph)
        # output_graph = self.relu(self.weight1 * hidden_states + self.weight2 * input_graph)
        # output_graph = self.relu(0.1 * hidden_states + 0.9 * input_graph)
        return output_graph

class ATT(nn.Module):
    def __init__(self, config_gat):
        super(ATT, self).__init__()
        layer = GATLayer(config_gat)
        self.encoder = nn.ModuleList([copy.deepcopy(layer) for _ in range(config_gat.num_layers)])

    def forward(self, input_graph):
        hidden_states = input_graph
        for layer_module in self.encoder:
            hidden_states = layer_module(hidden_states)
        return hidden_states  # B, seq_len, D