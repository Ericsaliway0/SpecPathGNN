import json
import networkx as nx
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import itertools
import scipy.sparse as sp
##from dgl.nn import GATConv
import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import edge_softmax
import dgl.function as fn
from dgl.base import DGLError
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from dgl.nn import SAGEConv, GATConv, GraphConv, GINConv, ChebConv
from torch_geometric.utils import dropout_edge, negative_sampling, remove_self_loops, add_self_loops
import math
from torch.nn import Parameter
import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import ChebConv

class ACGNN_lamda_added(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3, dropout=0.3, epsilon=1e-4):
        """
        Efficient Graph Convolutional Network (EGCN) with:
        - Chebyshev Polynomial Approximation (Adaptive)
        - Early Stopping for Chebyshev Expansion
        - Three Aggregation Terms for Better Expressivity

        Parameters:
        - in_feats: Input feature size
        - hidden_feats: Hidden layer feature size
        - out_feats: Output feature size
        - k: Maximum Chebyshev polynomial order
        - dropout: Dropout rate for regularization
        - epsilon: Early stopping tolerance for Chebyshev computation
        """
        super(ACGNN, self).__init__()
        self.k = k  # Maximum Chebyshev order
        self.dropout = dropout
        self.epsilon = epsilon  # Stopping threshold

        # Chebyshev Convolution Layers
        self.cheb1 = ChebConv(in_feats, hidden_feats, k)
        self.cheb2 = ChebConv(hidden_feats, hidden_feats, k)
        self.cheb3 = ChebConv(hidden_feats, hidden_feats, k)  # Third aggregation term

        # Fully Connected Layer (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

        # Batch Normalization
        self.norm = nn.BatchNorm1d(hidden_feats)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, g, features):
        """
        Forward pass for EGCN with adaptive Chebyshev polynomial expansion.

        Parameters:
        - g: DGL graph
        - features: Input node features

        Returns:
        - Output predictions after passing through EGCN layers
        """
        # Compute lambda_max for Chebyshev polynomials
        lambda_max = dgl.laplacian_lambda_max(g)

        # First Chebyshev Convolution
        x = F.relu(self.cheb1(g, features, lambda_max=lambda_max))
        x = self.norm(x)

        # Early stopping check for Chebyshev polynomials
        prev_x = x.clone()
        for _ in range(1, self.k):
            x_new = F.relu(self.cheb2(g, x, lambda_max=lambda_max))
            if torch.norm(x_new - prev_x) < self.epsilon:
                break  # Stop if change is small
            prev_x = x_new.clone()
        
        # Third Aggregation Term for Feature Enhancement
        x_res = x  # Residual Connection
        x = F.relu(self.cheb3(g, x, lambda_max=lambda_max))
        x = self.dropout_layer(x) + x_res  # Efficient residual connection

        return self.mlp(x)

class ACGNN_ori(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=2, dropout=0.3):
        """
        Speed-optimized Adaptive Chebyshev Graph Neural Network.
        
        Parameters:
        - in_feats: Input feature size
        - hidden_feats: Hidden layer feature size
        - out_feats: Output feature size
        - k: Chebyshev polynomial order (lower for speed)
        - dropout: Dropout rate
        """
        super(ACGNN, self).__init__()
        self.k = k  # Adaptive Chebyshev order
        self.dropout = dropout
        
        # Reduced ChebConv layers (only 2 instead of 3)
        self.cheb1 = ChebConv(in_feats, hidden_feats, k)
        self.cheb2 = ChebConv(hidden_feats, hidden_feats, k)
        
        # Fully Connected Layers
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )
        
        # Faster Normalization
        self.norm = nn.BatchNorm1d(hidden_feats)  # BatchNorm is faster than LayerNorm
        
        # Regularization
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, g, features):
        """
        Forward pass for Fast Adaptive ACGNN.
        
        Parameters:
        - g: DGL graph
        - features: Input node features
        
        Returns:
        - Output tensor after passing through Fast ACGNN layers
        """
        x = F.relu(self.cheb1(g, features))
        x = self.norm(x)  # BatchNorm improves stability
        
        x_res = x  # Residual Connection
        x = F.relu(self.cheb2(g, x))
        x = self.dropout_layer(x) + x_res  # Efficient Residual
        
        return self.mlp(x)

class ACGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3, dropout=0.3, epsilon=1e-4):
        """
        Efficient Graph Convolutional Network (EGCN) with:
        - Chebyshev Polynomial Approximation (Adaptive)
        - Early Stopping for Chebyshev Expansion
        - Three Aggregation Terms for Better Expressivity

        Parameters:
        - in_feats: Input feature size
        - hidden_feats: Hidden layer feature size
        - out_feats: Output feature size
        - k: Maximum Chebyshev polynomial order
        - dropout: Dropout rate for regularization
        - epsilon: Early stopping tolerance for Chebyshev computation
        """
        super(ACGNN, self).__init__()
        self.k = k  # Maximum Chebyshev order
        self.dropout = dropout
        self.epsilon = epsilon  # Stopping threshold

        # Chebyshev Convolution Layers
        self.cheb1 = ChebConv(in_feats, hidden_feats, k)
        self.cheb2 = ChebConv(hidden_feats, hidden_feats, k)
        self.cheb3 = ChebConv(hidden_feats, hidden_feats, k)  # Third aggregation term

        # Fully Connected Layer (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

        # Batch Normalization
        self.norm = nn.BatchNorm1d(hidden_feats)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, graph, features, lambda_max=None):
        if lambda_max is None:
            lambda_max = dgl.laplacian_lambda_max(graph)

        x = F.relu(self.cheb1(graph, features, lambda_max=lambda_max))
        x = self.norm(x)

        prev_x = x.clone()
        for _ in range(1, self.k):
            x_new = F.relu(self.cheb2(graph, x, lambda_max=lambda_max))
            if torch.norm(x_new - prev_x) < self.epsilon:
                break
            prev_x = x_new.clone()

        x_res = x
        x = F.relu(self.cheb3(graph, x, lambda_max=lambda_max))
        x = self.dropout_layer(x) + x_res

        return self.mlp(x)

class ACGNN_ig(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3, dropout=0.3, epsilon=1e-4):
        super(ACGNN, self).__init__()
        self.k = k
        self.dropout = dropout
        self.epsilon = epsilon

        self.cheb1 = ChebConv(in_feats, hidden_feats, k)
        self.cheb2 = ChebConv(hidden_feats, hidden_feats, k)
        self.cheb3 = ChebConv(hidden_feats, hidden_feats, k)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

        self.norm = nn.BatchNorm1d(hidden_feats)
        self.dropout_layer = nn.Dropout(dropout)

        self.graph = None  # This will store the graph

    def set_graph(self, g):
        """Set the graph for forward pass — needed for Captum"""
        self.graph = g

    def forward(self, features):
        if self.graph is None:
            raise ValueError("Graph is not set. Use set_graph(g) before calling forward(features).")


        g = self.graph
        print("📦 Inside ACGNN forward:")
        print("   - features.shape:", features.shape)
        print("   - graph.num_nodes():", g.num_nodes())

        x = F.relu(self.cheb1(g, features))
        x = self.norm(x)

        prev_x = x.clone()
        for _ in range(1, self.k):
            x_new = F.relu(self.cheb2(g, x))
            if torch.norm(x_new - prev_x) < self.epsilon:
                break
            prev_x = x_new.clone()

        x_res = x
        x = F.relu(self.cheb3(g, x))
        x = self.dropout_layer(x) + x_res

        return self.mlp(x).squeeze()

class HGDC(torch.nn.Module):
    def __init__(self, args, weights=[0.95, 0.90, 0.15, 0.10]):
        super().__init__()
        self.args = args
        in_channels = self.args.in_channels
        hidden_channels = self.args.hidden_channels
        self.linear1 = Linear(in_channels, hidden_channels)

        # 3 convolutional layers for the original network
        self.conv_k1_1 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k2_1 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k3_1 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)
        
        # 3 convolutional layers for the auxiliary network
        self.conv_k1_2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k2_2 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k3_2 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)

        self.linear_r0 = Linear(hidden_channels, 1)
        self.linear_r1 = Linear(2 * hidden_channels, 1)
        self.linear_r2 = Linear(2 * hidden_channels, 1)
        self.linear_r3 = Linear(2 * hidden_channels, 1)

        # Attention weights on outputs of different convolutional layers
        self.weight_r0 = torch.nn.Parameter(torch.Tensor([weights[0]]), requires_grad=True)
        self.weight_r1 = torch.nn.Parameter(torch.Tensor([weights[1]]), requires_grad=True)
        self.weight_r2 = torch.nn.Parameter(torch.Tensor([weights[2]]), requires_grad=True)
        self.weight_r3 = torch.nn.Parameter(torch.Tensor([weights[3]]), requires_grad=True)

    def forward(self, data):
        x_input = data.x
        edge_index_1 = data.edge_index
        edge_index_2 = data.edge_index_aux

        edge_index_1, _ = dropout_edge(edge_index_1, p=0.5, 
                                       force_undirected=True, 
                                       training=self.training)
        edge_index_2, _ = dropout_edge(edge_index_2, p=0.5, 
                                       force_undirected=True, 
                                       training=self.training)

        x_input = F.dropout(x_input, p=0.5, training=self.training)

        R0 = torch.relu(self.linear1(x_input))

        R_k1_1 = self.conv_k1_1(R0, edge_index_1)
        R_k1_2 = self.conv_k1_2(R0, edge_index_2)
        R1 = torch.cat((R_k1_1, R_k1_2), 1)

        R_k2_1 = self.conv_k2_1(R1, edge_index_1)
        R_k2_2 = self.conv_k2_2(R1, edge_index_2)
        R2 = torch.cat((R_k2_1, R_k2_2), 1)

        R_k3_1 = self.conv_k3_1(R2, edge_index_1)
        R_k3_2 = self.conv_k3_2(R2, edge_index_2)
        R3 = torch.cat((R_k3_1, R_k3_2), 1)

        R0 = F.dropout(R0, p=0.5, training=self.training)
        res0 = self.linear_r0(R0)
        R1 = F.dropout(R1, p=0.5, training=self.training)
        res1 = self.linear_r1(R1)
        R2 = F.dropout(R2, p=0.5, training=self.training)
        res2 = self.linear_r2(R2)
        R3 = F.dropout(R3, p=0.5, training=self.training)
        res3 = self.linear_r3(R3)

        out = res0 * self.weight_r0 + res1 * self.weight_r1 + res2 * self.weight_r2 + res3 * self.weight_r3
        return out

class MTGCN(torch.nn.Module):
    def __init__(self, args):
        super(MTGCN, self).__init__()
        self.args = args
        self.conv1 = ChebConv(58, 300, K=2, normalization="sym")
        self.conv2 = ChebConv(300, 100, K=2, normalization="sym")
        self.conv3 = ChebConv(100, 1, K=2, normalization="sym")

        self.lin1 = Linear(58, 100)
        self.lin2 = Linear(58, 100)

        self.c1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.c2 = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self, data):
        edge_index, _ = dropout_edge(data.edge_index, p=0.5,
                                     force_undirected=True,
                                     num_nodes=data.x.size()[0],
                                     training=self.training)
        E = data.edge_index
        pb, _ = remove_self_loops(data.edge_index)
        pb, _ = add_self_loops(pb)

        x0 = F.dropout(data.x, training=self.training)
        x = torch.relu(self.conv1(x0, edge_index))
        x = F.dropout(x, training=self.training)
        x1 = torch.relu(self.conv2(x, edge_index))

        x = x1 + torch.relu(self.lin1(x0))
        z = x1 + torch.relu(self.lin2(x0))

        pos_loss = -torch.log(torch.sigmoid((z[E[0]] * z[E[1]]).sum(dim=1)) + 1e-15).mean()

        neg_edge_index = negative_sampling(pb, data.num_nodes, data.num_edges)

        neg_loss = -torch.log(
            1 - torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)) + 1e-15).mean()

        r_loss = pos_loss + neg_loss

        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x, r_loss, self.c1, self.c2

class EMOGI(torch.nn.Module):
    def __init__(self,args):
        super(EMOGI, self).__init__()
        self.args = args
        self.conv1 = ChebConv(58, 300, K=2)
        self.conv2 = ChebConv(300, 100, K=2)
        self.conv3 = ChebConv(100, 1, K=2)

    def forward(self, data):
        edge_index = data.edge_index
        x = F.dropout(data.x, training=self.training)
        x = torch.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = torch.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x

class ChebNet(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3):
        """
        ChebNet implementation using DGL's ChebConv.
        
        Parameters:
        - in_feats: Number of input features.
        - hidden_feats: Number of hidden layer features.
        - out_feats: Number of output features.
        - k: Chebyshev polynomial order.
        """
        super(ChebNet, self).__init__()
        self.cheb1 = ChebConv(in_feats, hidden_feats, k)
        self.cheb2 = ChebConv(hidden_feats, hidden_feats, k)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features):
        """
        Forward pass for ChebNet.
        
        Parameters:
        - g: DGL graph.
        - features: Input features tensor.
        
        Returns:
        - Output tensor after passing through ChebNet layers.
        """
        x = F.relu(self.cheb1(g, features))
        x = F.relu(self.cheb2(g, x))
        return self.mlp(x)

def cheby(i,x):
    if i==0:
        return 1
    elif i==1:
        return x
    else:
        T0=1
        T1=x
        for ii in range(2,i+1):
            T2=2*x*T1-T0
            T0,T1=T1,T2
        return T2

class ChebNetII(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3, dropout=0.5):
        """
        Custom ChebNetII with adaptive Chebyshev filtering and MLP head.
        
        Parameters:
        - in_feats: Input feature dimension.
        - hidden_feats: Hidden layer dimension.
        - out_feats: Output dimension (e.g., number of classes).
        - k: Order of Chebyshev polynomials (K).
        - dropout: Dropout probability.
        """
        super(ChebNetII, self).__init__()
        self.k = k
        self.dropout = dropout

        # Adaptive Chebyshev coefficients (learnable)
        self.temp = Parameter(torch.Tensor(k + 1))
        self.reset_parameters()

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, out_feats)
        )

    def reset_parameters(self):
        self.temp.data.fill_(1.0)

    def compute_adaptive_coefficients(self):
        """
        Compute adaptive Chebyshev coefficients using self.temp.
        """
        coe_tmp = F.relu(self.temp)
        coe = coe_tmp.clone()

        for i in range(self.k + 1):
            coe[i] = coe_tmp[0] * cheby(i, math.cos((self.k + 0.5) * math.pi / (self.k + 1)))
            for j in range(1, self.k + 1):
                x_j = math.cos((self.k - j + 0.5) * math.pi / (self.k + 1))
                coe[i] += coe_tmp[j] * cheby(i, x_j)
            coe[i] = 2 * coe[i] / (self.k + 1)

        coe[0] = coe[0] / 2  # scale the first coefficient
        return coe

    def forward(self, x_list, st=0, end=0):
        """
        Forward pass.

        Parameters:
        - x_list: List of tensors [Tx_0, Tx_1, ..., Tx_K], each of shape (N, F)
        - st, end: Optional slicing indices for subsetting input.

        Returns:
        - Log-softmax predictions (N, C)
        """
        coe = self.compute_adaptive_coefficients()

        # Weighted sum of Chebyshev basis components
        out = coe[0] * x_list[0][st:end, :]
        for k in range(1, self.k + 1):
            out += coe[k] * x_list[k][st:end, :]

        return F.log_softmax(self.mlp(out), dim=1)


class FeatureAttention(nn.Module):
    def __init__(self, feat_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or max(16, feat_dim // 4)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim)
        )

    def forward(self, x):
        gates = torch.sigmoid(self.net(x))
        return x * gates

class MomentAggregator:
    @staticmethod
    def compute_moments(g, features, eps=1e-6):
        with g.local_scope():
            g.ndata['h'] = features
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh_mean'))
            neigh_mean = g.ndata.get('neigh_mean', torch.zeros_like(features))

            g.ndata['h2'] = features * features
            g.update_all(fn.copy_u('h2', 'm2'), fn.mean('m2', 'neigh_m2'))
            neigh_m2 = g.ndata.get('neigh_m2', torch.zeros_like(features))
            neigh_var = torch.clamp(neigh_m2 - neigh_mean * neigh_mean, min=0.0)

            g.ndata['h3'] = features * features * features
            g.update_all(fn.copy_u('h3', 'm3'), fn.mean('m3', 'neigh_m3'))
            neigh_m3 = g.ndata.get('neigh_m3', torch.zeros_like(features))
            neigh_skew = neigh_m3 - 3 * neigh_mean * neigh_m2 + 2 * neigh_mean.pow(3)
            denom = (neigh_var + eps).pow(1.5)
            neigh_skew = neigh_skew / (denom + eps)

            return neigh_mean, neigh_var, neigh_skew

class DMGNN(nn.Module):
    def __init__(
        self,
        in_feat_dim,
        hidden_dim,
        out_dim,
        heads=4,
        dropout=0.5,
        use_moments=('mean', 'var', 'skew'),
        use_feature_attn=True,
        remote_emb_dim=0
    ):
        super().__init__()
        self.use_moments = use_moments
        self.use_feature_attn = use_feature_attn
        self.remote_emb_dim = remote_emb_dim
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.heads = heads

        if use_feature_attn:
            self.feat_attn = FeatureAttention(in_feat_dim)

        moment_channels = 0
        if 'mean' in use_moments: moment_channels += 1
        if 'var' in use_moments: moment_channels += 1
        if 'skew' in use_moments: moment_channels += 1

        total_input = in_feat_dim * (1 + moment_channels) + remote_emb_dim

        self.moment_proj = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.gat1 = GATConv(hidden_dim, hidden_dim // heads, num_heads=heads,
                            feat_drop=dropout, attn_drop=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, num_heads=1,
                            feat_drop=dropout, attn_drop=dropout)

        self.res_proj = nn.Linear(hidden_dim, hidden_dim)
        self.agg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def mix_moment_embed(self, g, features):
        neigh_mean, neigh_var, neigh_skew = MomentAggregator.compute_moments(g, features)
        parts = [features]
        if 'mean' in self.use_moments: parts.append(neigh_mean)
        if 'var' in self.use_moments: parts.append(neigh_var)
        if 'skew' in self.use_moments: parts.append(neigh_skew)
        return torch.cat(parts, dim=1)

    def forward(self, g, features, remote_emb=None):
        if self.use_feature_attn:
            features = self.feat_attn(features)

        mixed = self.mix_moment_embed(g, features)

        if remote_emb is not None and self.remote_emb_dim > 0:
            mixed = torch.cat([mixed, remote_emb], dim=1)

        h = self.moment_proj(mixed)

        x = self.gat1(g, h)
        x = x.view(x.shape[0], -1)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x2 = self.gat2(g, x).squeeze(1)
        x2 = F.elu(x2)

        if x.shape[1] != x2.shape[1]:
            x = self.res_proj(x)
        agg = self.agg_mlp(torch.cat([x, x2], dim=1))

        logits = self.classifier(agg)
        return logits  # ready for BCEWithLogitsLoss

class GIN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GIN, self).__init__()
        # Define the first GIN layer
        self.gin1 = GINConv(
            nn.Sequential(
                nn.Linear(in_feats, hidden_feats),
                nn.ReLU(),
                nn.Linear(hidden_feats, hidden_feats)
            ),
            'mean'  # Aggregation method: 'mean', 'max', or 'sum'
        )
        # Define the second GIN layer
        self.gin2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_feats, hidden_feats),
                nn.ReLU(),
                nn.Linear(hidden_feats, hidden_feats)
            ),
            'mean'
        )
        # MLP for final predictions
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features):
        # Apply the first GIN layer
        x = F.relu(self.gin1(g, features))
        # Apply the second GIN layer
        x = F.relu(self.gin2(g, x))
        # Apply the MLP
        return self.mlp(x)

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(in_feats, hidden_feats, aggregator_type='mean')
        self.sage2 = SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean')
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features):
        x = F.relu(self.sage1(g, features))
        x = F.relu(self.sage2(g, x))
        return self.mlp(x)

class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads=3):
        """
        Graph Attention Network (GAT).
        
        Parameters:
        - in_feats: Number of input features.
        - hidden_feats: Number of hidden layer features.
        - out_feats: Number of output features.
        - num_heads: Number of attention heads.
        """
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_feats, hidden_feats, num_heads, activation=F.relu)
        self.gat2 = GATConv(hidden_feats * num_heads, hidden_feats, num_heads, activation=F.relu)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats * num_heads, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features):
        """
        Forward pass for GAT.
        
        Parameters:
        - g: DGL graph.
        - features: Input features tensor.
        
        Returns:
        - Output tensor after passing through GAT layers.
        """
        x = self.gat1(g, features)
        x = x.flatten(1)  # Flatten the output of multi-head attention
        x = self.gat2(g, x)
        x = x.flatten(1)  # Flatten the output again
        return self.mlp(x)

class GAT_relaevance(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads=3):
        """
        Graph Attention Network (GAT) with attention weight extraction.
        
        Parameters:
        - in_feats: Number of input features.
        - hidden_feats: Number of hidden layer features.
        - out_feats: Number of output features.
        - num_heads: Number of attention heads.
        """
        super(GAT, self).__init__()
        self.num_heads = num_heads
        self.gat1 = GATConv(in_feats, hidden_feats, num_heads, activation=F.relu)
        self.gat2 = GATConv(hidden_feats * num_heads, hidden_feats, num_heads, activation=F.relu)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats * num_heads, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features, return_attention=False):
        """
        Forward pass with optional attention weight return.

        Parameters:
        - g: DGL graph.
        - features: Node feature tensor.
        - return_attention: If True, returns attention weights.

        Returns:
        - Output tensor (and attention weights if requested).
        """
        if return_attention:
            x, attn1 = self.gat1(g, features, get_attention=True)
            x = x.flatten(1)
            x, attn2 = self.gat2(g, x, get_attention=True)
            x = x.flatten(1)
            out = self.mlp(x)
            return out, (attn1, attn2)
        else:
            x = self.gat1(g, features)
            x = x.flatten(1)
            x = self.gat2(g, x)
            x = x.flatten(1)
            return self.mlp(x)

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCN, self).__init__()
        self.gcn1 = GraphConv(in_feats, hidden_feats)
        self.gcn2 = GraphConv(hidden_feats, hidden_feats)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features):
        x = F.relu(self.gcn1(g, features))
        x = F.relu(self.gcn2(g, x))
        return self.mlp(x)

class GIN_lrp_x(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, lrp_rule='epsilon'):
        super().__init__()
        self.lrp_rule = lrp_rule
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

        self.fc1 = nn.Linear(in_feats, hidden_feats)
        self.fc2 = nn.Linear(hidden_feats, hidden_feats)
        self.fc3 = nn.Linear(hidden_feats, hidden_feats)
        self.out_fc1 = nn.Linear(hidden_feats, hidden_feats)
        self.out_fc2 = nn.Linear(hidden_feats, out_feats)

        self.gin1 = GINConv(nn.Sequential(self.fc1, self.act1, self.fc2), aggregator_type='mean')
        self.gin2 = GINConv(nn.Sequential(self.fc3, self.act2), aggregator_type='mean')

        # Cache for forward pass
        self.cache = {}

    def forward(self, g, x):
        self.cache.clear()

        # Apply first GIN layer
        x_input = x.clone()
        x1 = self.fc1(x_input)
        self.cache['x1'] = x_input
        x = self.act1(x1)

        x2 = self.fc2(x)
        self.cache['x2'] = x2
        x = self.gin1(g, x2)  # Feed x2 to gin1, not reusing x2 again after gin1

        # Apply second GIN layer
        x3 = self.fc3(x)
        self.cache['x3'] = x3
        x = self.act2(x3)
        x = self.gin2(g, x)

        # Output MLP
        out1 = self.out_fc1(x)
        self.cache['out1'] = out1
        x = F.relu(out1)
        out2 = self.out_fc2(x)

        self.cache['x_out'] = x
        return out2


        def relprop(self, R, method='epsilon', epsilon=1e-6):
            """
            R: Relevance scores from output [batch_size, out_feats]
            Returns: Relevance scores per input feature [batch_size, in_feats]
            """
            if method == 'zplus':
                rule = lrp_linear_zplus
            else:
                rule = lambda i, w, o, r: lrp_linear_eps(i, w, o, r, epsilon=epsilon)

            # Output layer
            out_fc2_input = self.cache['x_out']
            R = rule(out_fc2_input, self.out_fc2.weight, None, R)

            # Output MLP
            R = rule(self.cache['out1'], self.out_fc1.weight, None, R)

            # GIN2 and act
            R = rule(self.cache['x3'], self.fc3.weight, None, R)

            # GIN1
            R = rule(self.cache['x2'], self.fc2.weight, None, R)

            # Input
            R = rule(self.cache['x1'], self.fc1.weight, None, R)

            return R

    def relprop(self, R, method='epsilon', epsilon=1e-6):
        """
        R: Relevance scores from output [batch_size, out_feats]
        Returns: Relevance scores per input feature [batch_size, in_feats]
        """
        if method == 'zplus':
            rule = lrp_linear_zplus
        else:
            rule = lambda i, w, o, r: lrp_linear_eps(i, w, o, r, epsilon=epsilon)

        # Output layer
        out_fc2_input = self.cache['x_out']
        R = rule(out_fc2_input, self.out_fc2.weight, None, R)

        # Output MLP
        R = rule(self.cache['out1'], self.out_fc1.weight, None, R)

        # GIN2 and act
        R = rule(self.cache['x3'], self.fc3.weight, None, R)

        # GIN1
        R = rule(self.cache['x2'], self.fc2.weight, None, R)

        # Input
        R = rule(self.cache['x1'], self.fc1.weight, None, R)

        return R

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ##bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # Ensure targets are of type float
        targets = targets.float()

        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        probas = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probas, 1 - probas)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


#################################################
## _return_embeddings
#################################################


# class ACGNN(nn.Module):
#     def __init__(self, in_feats, hidden_feats, out_feats, k=3, dropout=0.3, epsilon=1e-4):
#         super(ACGNN, self).__init__()
#         self.k = k
#         self.dropout = dropout
#         self.epsilon = epsilon

#         self.cheb1 = ChebConv(in_feats, hidden_feats, k)
#         self.cheb2 = ChebConv(hidden_feats, hidden_feats, k)
#         self.cheb3 = ChebConv(hidden_feats, hidden_feats, k)

#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_feats, hidden_feats),
#             nn.ReLU(),
#             nn.Linear(hidden_feats, out_feats)
#         )

#         self.norm = nn.BatchNorm1d(hidden_feats)
#         self.dropout_layer = nn.Dropout(dropout)

#     def forward(self, graph, features, lambda_max=None, return_embeddings=False):
#         if lambda_max is None:
#             lambda_max = dgl.laplacian_lambda_max(graph)

#         x = F.relu(self.cheb1(graph, features, lambda_max=lambda_max))
#         x = self.norm(x)

#         prev_x = x.clone()
#         for _ in range(1, self.k):
#             x_new = F.relu(self.cheb2(graph, x, lambda_max=lambda_max))
#             if torch.norm(x_new - prev_x) < self.epsilon:
#                 break
#             prev_x = x_new.clone()

#         x_res = x
#         x = F.relu(self.cheb3(graph, x, lambda_max=lambda_max))
#         x = self.dropout_layer(x) + x_res

#         if return_embeddings:
#             return x  # Shape: [num_nodes, hidden_feats]

#         return self.mlp(x)

# class GIN(nn.Module):
#     def __init__(self, in_feats, hidden_feats, out_feats):
#         super(GIN, self).__init__()
#         # Define the first GIN layer
#         self.gin1 = GINConv(
#             nn.Sequential(
#                 nn.Linear(in_feats, hidden_feats),
#                 nn.ReLU(),
#                 nn.Linear(hidden_feats, hidden_feats)
#             ),
#             'mean'
#         )
#         # Define the second GIN layer
#         self.gin2 = GINConv(
#             nn.Sequential(
#                 nn.Linear(hidden_feats, hidden_feats),
#                 nn.ReLU(),
#                 nn.Linear(hidden_feats, hidden_feats)
#             ),
#             'mean'
#         )
#         # MLP for final predictions
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_feats, hidden_feats),
#             nn.ReLU(),
#             nn.Linear(hidden_feats, out_feats)
#         )

#     def forward(self, g, features, return_embeddings=False):
#         # Apply the first and second GIN layers
#         x = F.relu(self.gin1(g, features))
#         x = F.relu(self.gin2(g, x))

#         if return_embeddings:
#             return x  # shape: [num_nodes, hidden_feats]

#         return self.mlp(x)

# class GCN(nn.Module):
#     def __init__(self, in_feats, hidden_feats, out_feats):
#         super(GCN, self).__init__()
#         self.gcn1 = GraphConv(in_feats, hidden_feats)
#         self.gcn2 = GraphConv(hidden_feats, hidden_feats)
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_feats, hidden_feats),
#             nn.ReLU(),
#             nn.Linear(hidden_feats, out_feats)
#         )

#     def forward(self, g, features, return_embeddings=False):
#         x = F.relu(self.gcn1(g, features))
#         x = F.relu(self.gcn2(g, x))

#         if return_embeddings:
#             return x  # shape: [num_nodes, hidden_feats]

#         return self.mlp(x)

# class GraphSAGE(nn.Module):
#     def __init__(self, in_feats, hidden_feats, out_feats):
#         super(GraphSAGE, self).__init__()
#         self.sage1 = SAGEConv(in_feats, hidden_feats, aggregator_type='mean')
#         self.sage2 = SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean')
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_feats, hidden_feats),
#             nn.ReLU(),
#             nn.Linear(hidden_feats, out_feats)
#         )

#     def forward(self, g, features, return_embeddings=False):
#         x = F.relu(self.sage1(g, features))
#         x = F.relu(self.sage2(g, x))

#         if return_embeddings:
#             return x  # shape: [num_nodes, hidden_feats]

#         return self.mlp(x)

# class GAT(nn.Module):
#     def __init__(self, in_feats, hidden_feats, out_feats, num_heads=3):
#         """
#         Graph Attention Network (GAT).

#         Parameters:
#         - in_feats: Number of input features.
#         - hidden_feats: Number of hidden layer features.
#         - out_feats: Number of output features.
#         - num_heads: Number of attention heads.
#         """
#         super(GAT, self).__init__()
#         self.gat1 = GATConv(in_feats, hidden_feats, num_heads, activation=F.relu)
#         self.gat2 = GATConv(hidden_feats * num_heads, hidden_feats, num_heads, activation=F.relu)
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_feats * num_heads, hidden_feats),
#             nn.ReLU(),
#             nn.Linear(hidden_feats, out_feats)
#         )

#     def forward(self, g, features, return_embeddings=False):
#         """
#         Forward pass for GAT.

#         Parameters:
#         - g: DGL graph.
#         - features: Input features tensor.
#         - return_embeddings: If True, return node embeddings before MLP.

#         Returns:
#         - Either the output tensor after the MLP or intermediate node embeddings.
#         """
#         x = self.gat1(g, features)
#         x = x.flatten(1)  # Flatten the output of multi-head attention
#         x = self.gat2(g, x)
#         x = x.flatten(1)  # Flatten the output again

#         if return_embeddings:
#             return x  # Shape: [num_nodes, hidden_feats * num_heads]

#         return self.mlp(x)

# class ChebNet(nn.Module):
#     def __init__(self, in_feats, hidden_feats, out_feats, k=3):
#         """
#         ChebNet implementation using DGL's ChebConv.

#         Parameters:
#         - in_feats: Number of input features.
#         - hidden_feats: Number of hidden layer features.
#         - out_feats: Number of output features.
#         - k: Chebyshev polynomial order.
#         """
#         super(ChebNet, self).__init__()
#         self.cheb1 = ChebConv(in_feats, hidden_feats, k)
#         self.cheb2 = ChebConv(hidden_feats, hidden_feats, k)
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_feats, hidden_feats),
#             nn.ReLU(),
#             nn.Linear(hidden_feats, out_feats)
#         )

#     def forward(self, g, features, return_embeddings=False):
#         """
#         Forward pass for ChebNet.

#         Parameters:
#         - g: DGL graph.
#         - features: Input features tensor.
#         - return_embeddings: If True, return node embeddings before MLP.

#         Returns:
#         - Either the output tensor after the MLP or intermediate node embeddings.
#         """
#         x = F.relu(self.cheb1(g, features))
#         x = F.relu(self.cheb2(g, x))

#         if return_embeddings:
#             return x  # Shape: [num_nodes, hidden_feats]

#         return self.mlp(x)
    

# Define the Focal Loss class
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss  

class MLPPredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(input_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, 1)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

class LinkPredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LinkPredictor, self).__init__()
        self.W1 = nn.Linear(input_size * 2, hidden_size)
        self.W2 = nn.Linear(hidden_size, 1)

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(lambda edges: {'score': self.W2(F.relu(self.W1(torch.cat([edges.src['h'], edges.dst['h']], dim=1)))).squeeze(1)})
            return g.edata['score']

class GATConv(nn.Module):
    def __init__(self,
                 in_feats: Union[int, Tuple[int, int]],
                 out_feats: int,
                 num_heads: int,
                 feat_drop: float = 0.,
                 attn_drop: float = 0.,
                 negative_slope: float = 0.2,
                 residual: bool = False,
                 activation: Optional[Callable] = None,
                 allow_zero_in_degree: bool = False,
                 bias: bool = True) -> None:
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = dgl.utils.expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)

        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.residual = residual
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer("res_fc", None)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation
        
        # Add normalization layer
        self.norm = nn.BatchNorm1d(num_heads * out_feats)

    def reset_parameters(self) -> None:
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.res_fc is not None and not isinstance(self.res_fc, nn.Identity):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value: bool) -> None:
        """Set the flag to allow zero in-degree for the graph."""
        self._allow_zero_in_degree = set_value

    def forward(self, graph: DGLGraph, feat: Union[Tensor, Tuple[Tensor, Tensor]]) -> Tensor:
        """Forward computation."""
        with graph.local_scope():
            if not self._allow_zero_in_degree and (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting `allow_zero_in_degree` '
                               'to `True` when constructing this module will '
                               'suppress this check and let the users handle '
                               'it by themselves.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if hasattr(self, 'fc_src'):
                    feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)

            graph.srcdata.update({'ft': feat_src, 'el': (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)})
            graph.dstdata.update({'er': (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))

            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], self._num_heads, self._out_feats)
                rst = rst + resval

            if self.bias is not None:
                rst = rst + self.bias.view(1, -1, self._out_feats)

            # Apply normalization
            rst = rst.view(rst.shape[0], -1)
            rst = self.norm(rst)
            rst = rst.view(rst.shape[0], self._num_heads, self._out_feats)

            if self.activation:
                rst = self.activation(rst)

            return rst


class GATModel(nn.Module):
    def __init__(self, in_feats, out_feats, num_layers=1, num_heads=4, feat_drop=0.0, attn_drop=0.0, do_train=False):
        super(GATModel, self).__init__()
        self.do_train = do_train
        
        assert out_feats % num_heads == 0, "out_feats must be divisible by num_heads"
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(GATConv(in_feats, out_feats // num_heads, num_heads, feat_drop=feat_drop, attn_drop=attn_drop, residual=True, activation=F.leaky_relu, allow_zero_in_degree=True))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(out_feats, out_feats // num_heads, num_heads, feat_drop=feat_drop, attn_drop=attn_drop, residual=True, activation=F.leaky_relu, allow_zero_in_degree=True))
        
        self.predict = nn.Linear(out_feats, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = self.dropout(h)
            h = layer(g, h).flatten(1)
            h = self.leaky_relu(h)  # Apply LeakyReLU activation
        
        if not self.do_train:
            return h.detach()
        
        logits = self.predict(h)
        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv

class GATModel(nn.Module):
    def __init__(self, in_feats, out_feats, num_layers=1, num_heads=4,
                 feat_drop=0.0, attn_drop=0.0, do_train=False):
        super(GATModel, self).__init__()
        self.do_train = do_train
        
        assert out_feats % num_heads == 0, "out_feats must be divisible by num_heads"
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(
            GATConv(
                in_feats, out_feats // num_heads, num_heads,
                feat_drop=feat_drop, attn_drop=attn_drop,
                residual=True, activation=F.leaky_relu,
                allow_zero_in_degree=True
            )
        )
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(
                GATConv(
                    out_feats, out_feats // num_heads, num_heads,
                    feat_drop=feat_drop, attn_drop=attn_drop,
                    residual=True, activation=F.leaky_relu,
                    allow_zero_in_degree=True
                )
            )
        
        self.predict = nn.Linear(out_feats, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = self.dropout(h)
            h = layer(g, h).flatten(1)
            h = self.leaky_relu(h)
        
        if self.do_train:  # explicit control
            logits = self.predict(h)
            return logits
        else:  # inference / embeddings only
            return h.detach()

class GATModel(nn.Module):
    def __init__(self, in_feats, out_feats, num_layers=1, num_heads=4,
                 feat_drop=0.0, attn_drop=0.0, do_train=False):
        super(GATModel, self).__init__()
        self.do_train = do_train
        
        assert out_feats % num_heads == 0, "out_feats must be divisible by num_heads"
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(
            GATConv(
                in_feats, out_feats // num_heads, num_heads,
                feat_drop=feat_drop, attn_drop=attn_drop,
                residual=True, activation=F.leaky_relu,
                allow_zero_in_degree=True
            )
        )
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(
                GATConv(
                    out_feats, out_feats // num_heads, num_heads,
                    feat_drop=feat_drop, attn_drop=attn_drop,
                    residual=True, activation=F.leaky_relu,
                    allow_zero_in_degree=True
                )
            )
        
        self.predict = nn.Linear(out_feats, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=0.3)

        
    def forward(self, g, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(g, h).flatten(1) if i < len(self.layers) - 1 else layer(g, h).mean(1)
        return h

class GATModel(nn.Module):
    def __init__(self, in_feats, hidden_feats=128, out_feats=64,
                num_layers=1, num_heads=4,
                feat_drop=0.0, attn_drop=0.0, do_train=False):

        super(GATModel, self).__init__()  # ✅ must come first

        self.layers = nn.ModuleList()
        # input -> hidden
        self.layers.append(dgl.nn.GATConv(
            in_feats, hidden_feats, num_heads,
            feat_drop=feat_drop, attn_drop=attn_drop,
            activation=F.elu
        ))
        
        # hidden -> hidden (num_layers - 2)
        for _ in range(num_layers - 2):
            self.layers.append(dgl.nn.GATConv(
                hidden_feats * num_heads, hidden_feats, num_heads,
                feat_drop=feat_drop, attn_drop=attn_drop,
                activation=F.elu
            ))
        
        # last hidden -> output
        self.layers.append(dgl.nn.GATConv(
            hidden_feats * num_heads, out_feats, num_heads,
            feat_drop=feat_drop, attn_drop=attn_drop,
            activation=None
        ))

    def forward(self, g, x, return_logits: bool = True):
        h = x
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                h = layer(g, h).flatten(1)   # flatten heads
            else:
                h = layer(g, h).mean(1)      # average heads -> logits shape: [N, out_feats]

        if return_logits:
            return h                        # always shape [N, out_feats]
        else:
            return F.log_softmax(h, dim=-1) # optional for inference


class GATModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats,
                 num_layers=1, num_heads=4,
                 feat_drop=0.0, attn_drop=0.0, do_train=False):
        super(GATModel, self).__init__()

        self.layers = nn.ModuleList()
        # input -> hidden
        self.layers.append(dgl.nn.GATConv(
            in_feats, hidden_feats, num_heads,
            feat_drop=feat_drop, attn_drop=attn_drop,
            activation=F.elu,
            allow_zero_in_degree=True   # ✅ fix
        ))
        
        # hidden -> hidden (num_layers - 2)
        for _ in range(num_layers - 2):
            self.layers.append(dgl.nn.GATConv(
                hidden_feats * num_heads, hidden_feats, num_heads,
                feat_drop=feat_drop, attn_drop=attn_drop,
                activation=F.elu,
                allow_zero_in_degree=True   # ✅ fix
            ))
        
        # last hidden -> output
        self.layers.append(dgl.nn.GATConv(
            hidden_feats * num_heads, out_feats, num_heads,
            feat_drop=feat_drop, attn_drop=attn_drop,
            activation=None,
            allow_zero_in_degree=True   # ✅ fix
        ))

    def forward(self, g, x, return_logits: bool = True):
        h = x
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                h = layer(g, h).flatten(1)   # flatten heads
            else:
                h = layer(g, h).mean(1)      # average heads -> logits shape: [N, out_feats]

        if return_logits:
            return h                        # always shape [N, out_feats]
        else:
            return F.log_softmax(h, dim=-1) # optional for inference



class ECGNN(nn.Module):
    """
    Efficient Chebyshev Graph Convolutional Network (ECGNN).

    - Uses Chebyshev polynomial filters with early stopping
    - Stack of ChebConv layers
    - Dropout + activation applied between layers
    """

    def __init__(self, in_feats, hidden_feats, out_feats, k=3, dropout=0.3, epsilon=1e-4):
        super(ECGNN, self).__init__()

        self.k = k
        self.dropout = dropout
        self.epsilon = epsilon

        # Define sequential ChebConv layers
        self.layers = nn.ModuleList()

        # Input → Hidden
        self.layers.append(ChebConv(in_feats, hidden_feats, k=self.k))

        # Hidden → Hidden
        self.layers.append(ChebConv(hidden_feats, hidden_feats, k=self.k))

        # Hidden → Output
        self.layers.append(ChebConv(hidden_feats, out_feats, k=self.k))

    def forward(self, g, x, return_logits: bool = True):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(g, h)

            # Only apply activation & dropout on hidden layers
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        if return_logits:
            return h  # logits
        else:
            return F.softmax(h, dim=-1)  # normalized prediction scores

class ECGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3, dropout=0.3, epsilon=1e-4):
        super(ECGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ChebConv(in_feats, hidden_feats, k))
        self.layers.append(ChebConv(hidden_feats, out_feats, k))
        self.dropout = nn.Dropout(dropout)
        self.epsilon = epsilon

    def forward(self, graph=None, feat=None, **kwargs):
        # match GNNExplainer’s call
        h = feat
        for conv in self.layers:
            h = conv(graph, h)
            h = F.relu(h)
            h = self.dropout(h)
        return h

class ECGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ChebConv(in_feats, hidden_feats, k))
        self.layers.append(ChebConv(hidden_feats, out_feats, k))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x, return_logits_only=True):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        logits = h
        return logits if return_logits_only else (logits, h)


import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import ChebConv

class ECGNN_ori(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3, dropout=0.3, epsilon=1e-4):
        super(ECGNN, self).__init__()
        # Save dimensions manually (ChebConv does not expose them)
        self.in_feats = in_feats
        self.hidden_feats = hidden_feats
        self.out_feats = out_feats

        self.conv1 = ChebConv(in_feats, hidden_feats, k)
        self.conv2 = ChebConv(hidden_feats, out_feats, k)

        self.dropout = nn.Dropout(dropout)
        self.epsilon = epsilon

    def forward(self, graph, feat, eweight=None):
        # ---- Layer 1 ----
        h = self.conv1(graph, feat)  # ChebConv doesn’t use edge weights by default
        h = F.relu(h)
        h = self.dropout(h)

        # ---- Layer 2 ----
        h = self.conv2(graph, h)
        return h

import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import ChebConv

class ECGNN(nn.Module):
    def __init__(self, graph, in_feats, hidden_feats, out_feats, k=3, dropout=0.3):
        super(ECGNN, self).__init__()

        # ---- Compute lambda_max once ----
        lambda_max = dgl.laplacian_lambda_max(graph)

        # ---- Save dims ----
        self.in_feats = in_feats
        self.hidden_feats = hidden_feats
        self.out_feats = out_feats

        # ---- Pass lambda_max explicitly ----
        self.conv1 = ChebConv(in_feats, hidden_feats, k, lambda_max=lambda_max)
        self.conv2 = ChebConv(hidden_feats, out_feats, k, lambda_max=lambda_max)

        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, feat, eweight=None):
        h = self.conv1(graph, feat)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.conv2(graph, h)
        return h
