import json
import networkx as nx
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import itertools
import scipy.sparse as sp
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
from dgl.nn import GraphConv
from dgl.nn import ChebConv
from dgl.nn import GINConv
from dgl import laplacian_lambda_max

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

class GCNModel(nn.Module):
    def __init__(self, in_feats, dim_latent, num_layers=2, do_train=False):
        super().__init__()
        self.do_train = do_train
        self.conv_0 = GraphConv(in_feats=in_feats, out_feats=dim_latent)
        
        self.relu = nn.LeakyReLU()
        self.layers = nn.ModuleList([GraphConv(in_feats=dim_latent, out_feats=dim_latent) for _ in range(num_layers - 1)])
        self.predict = nn.Linear(dim_latent, 1)

    def forward(self, graph, features):
        graph = dgl.add_self_loop(graph)
        embedding = self.conv_0(graph, features)

        for conv in self.layers:
            embedding = self.relu(embedding)
            ##print('graph-----------------\n', graph)
            ##print('embedding===============\n', embedding)
            embedding = conv(graph, embedding)
        
        if not self.do_train:
            return embedding.detach()
        
        logits = self.predict(embedding)
        return logits

class MLPPredictor_ori(nn.Module):
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

class ChebNetModel_ori(nn.Module):
    def __init__(self, graph, in_feats, dim_latent, num_layers=2, k=3, dropout=0.3, do_train=True):
        super().__init__()

        self.graph = graph
        self.do_train = do_train
        self.k = k
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()

        # First layer
        self.conv_0 = ChebConv(in_feats, dim_latent, k)

        # Additional layers
        self.layers = nn.ModuleList([
            ChebConv(dim_latent, dim_latent, k)
            for _ in range(num_layers - 1)
        ])

        # Predictor used for link prediction
        self.predict = nn.Linear(dim_latent, 1)

    def forward(self, graph, features):
        h = self.conv_0(graph, features)
        h = self.relu(h)

        for conv in self.layers:
            h = conv(graph, h)
            h = self.relu(h)
            h = self.dropout(h)

        if not self.do_train:
            return h.detach()

        logits = self.predict(h)
        return logits

class ChebNetModel(nn.Module):
    def __init__(self, graph, in_feats, dim_latent, num_layers=2, k=3, dropout=0.3, do_train=True):
        super().__init__()

        self.graph = graph
        self.do_train = do_train
        self.k = k
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()

        # Precompute lambda_max ONCE
        self.lambda_max = laplacian_lambda_max(graph)

        # First Chebyshev convolution
        self.conv_0 = ChebConv(in_feats, dim_latent, k)

        # Additional layers
        self.layers = nn.ModuleList([
            ChebConv(dim_latent, dim_latent, k)
            for _ in range(num_layers - 1)
        ])

        # Link prediction head
        self.predict = nn.Linear(dim_latent, 1)

    def forward(self, graph, features):
        # Pass lambda_max explicitly (DGL 2.x requirement)
        h = self.conv_0(graph, features, self.lambda_max)
        h = self.relu(h)

        for conv in self.layers:
            h = conv(graph, h, self.lambda_max)
            h = self.relu(h)
            h = self.dropout(h)

        # If only encoding (e.g., inference step)
        if not self.do_train:
            return h.detach()

        # Link prediction logits
        logits = self.predict(h)
        return logits

class GINNetModel(nn.Module):
    def __init__(
        self,
        graph,
        in_feats,
        dim_latent,
        num_layers=2,
        dropout=0.3,
        do_train=True
    ):
        super().__init__()

        self.graph = graph
        self.do_train = do_train
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()

        # ----------------------------
        # GIN layers
        # ----------------------------
        def gin_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )

        self.conv_0 = GINConv(
            gin_mlp(in_feats, dim_latent),
            aggregator_type="sum"
        )

        self.layers = nn.ModuleList([
            GINConv(
                gin_mlp(dim_latent, dim_latent),
                aggregator_type="sum"
            )
            for _ in range(num_layers - 1)
        ])

        # ----------------------------
        # Link prediction head
        # ----------------------------
        self.predict = nn.Linear(dim_latent, 1)

    def forward(self, graph, features):
        h = self.conv_0(graph, features)
        h = self.relu(h)

        for conv in self.layers:
            h = conv(graph, h)
            h = self.relu(h)
            h = self.dropout(h)

        # Encoder-only mode
        if not self.do_train:
            return h.detach()

        logits = self.predict(h)
        return logits

class MLPPredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        input_size = 2 * node_embedding_dim
        """
        super().__init__()
        self.W1 = nn.Linear(input_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, 1)

    def apply_edges(self, edges):
        """
        Used during standard DGL link prediction
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        return {
            'score': self.W2(F.relu(self.W1(h))).squeeze(1)
        }

    def forward(self, g, h):
        """
        Standard forward pass with graph
        """
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

    def forward_from_embedding(self, edge_embed):
        """
        Forward pass directly from edge embeddings
        (used for Integrated Gradients)

        edge_embed: Tensor [E × (2 * node_embedding_dim)]
        """
        x = F.relu(self.W1(edge_embed))
        return self.W2(x).squeeze(1)

class MLPPredictor_(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def apply_edges(self, edges):
        h = torch.cat(
            [edges.src['h'], edges.dst['h']],
            dim=1
        )
        return {'score': self.mlp(h)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']
