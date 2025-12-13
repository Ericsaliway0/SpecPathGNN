import json
import os
from matplotlib import pyplot as plt
import torch
import itertools
import dgl
import numpy as np
import scipy.sparse as sp
from dgl.dataloading import GraphDataLoader
from .models import LinkPredictor, GATModel, MLPPredictor, FocalLoss
from .utils import (plot_scores, compute_hits_k, compute_auc, compute_f1, compute_focalloss,
                    compute_accuracy, compute_precision, compute_recall, compute_map,
                    compute_focalloss_with_symmetrical_confidence, compute_auc_with_symmetrical_confidence,
                    compute_f1_with_symmetrical_confidence, compute_accuracy_with_symmetrical_confidence,
                    compute_precision_with_symmetrical_confidence, compute_recall_with_symmetrical_confidence,
                    compute_map_with_symmetrical_confidence)
from scipy.stats import sem
from torch.optim.lr_scheduler import StepLR,ExponentialLR

import networkx as nx
import community  # python-louvain
import numpy as np
import os
import json
import torch
import itertools
import dgl
import scipy.sparse as sp
import networkx as nx
import community  # python-louvain
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR

def louvain_clustering_top_predictions(G_dgl, pred_scores, top_k=1000):
    """
    Perform Louvain clustering on the top-k predicted edges.
    
    Arguments:
    - G_dgl: DGLGraph (original full graph)
    - pred_scores: torch.Tensor of predicted scores for all edges (concatenated pos+neg)
    - top_k: int, number of top predicted edges to use
    
    Returns:
    - partition: dict mapping node -> cluster_id
    - subgraph: nx.Graph of top-k edges
    """
    # Convert to CPU numpy array
    scores = pred_scores.detach().cpu().numpy()

    # Sort edges by predicted score
    sorted_idx = np.argsort(-scores)  # descending order
    top_idx = sorted_idx[:top_k]

    # Get original edges u,v from DGL
    u, v = G_dgl.edges()
    u = u.cpu().numpy()
    v = v.cpu().numpy()
    
    top_u = u[top_idx]
    top_v = v[top_idx]

    # Build NetworkX subgraph
    subgraph = nx.Graph()
    for src, dst in zip(top_u, top_v):
        subgraph.add_edge(src, dst)

    # Louvain clustering
    partition = community.best_partition(subgraph)  # node -> cluster_id

    print(f"Louvain detected {len(set(partition.values()))} clusters")
    return partition, subgraph

def train_and_evaluate(args, G_dgl, node_features):
    u, v = G_dgl.edges()
    eids = np.arange(G_dgl.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    val_size = int(len(eids) * 0.1)
    train_size = G_dgl.number_of_edges() - test_size - val_size

    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    val_pos_u, val_pos_v = u[eids[test_size:test_size + val_size]], v[eids[test_size:test_size + val_size]]
    train_pos_u, train_pos_v = u[eids[test_size + val_size:]], v[eids[test_size + val_size:]]

    ##adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(G_dgl.number_of_nodes(), G_dgl.number_of_nodes()))
    adj_neg = 1 - adj.todense() - np.eye(G_dgl.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), G_dgl.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    val_neg_u, val_neg_v = neg_u[neg_eids[test_size:test_size + val_size]], neg_v[neg_eids[test_size:test_size + val_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size + val_size:]], neg_v[neg_eids[test_size + val_size:]]

    train_g = dgl.remove_edges(G_dgl, eids[:test_size + val_size])


    def create_graph(u, v, num_nodes, full_node_features):
        g = dgl.graph((u, v), num_nodes=num_nodes)
        # Copy node features from full graph
        g.ndata['feat'] = full_node_features
        return g

    train_pos_g = create_graph(train_pos_u, train_pos_v, G_dgl.number_of_nodes())
    train_neg_g = create_graph(train_neg_u, train_neg_v, G_dgl.number_of_nodes())
    val_pos_g = create_graph(val_pos_u, val_pos_v, G_dgl.number_of_nodes())
    val_neg_g = create_graph(val_neg_u, val_neg_v, G_dgl.number_of_nodes())
    test_pos_g = create_graph(test_pos_u, test_pos_v, G_dgl.number_of_nodes())
    test_neg_g = create_graph(test_neg_u, test_neg_v, G_dgl.number_of_nodes())

    model = GATModel(
        node_features.shape[1], 
        out_feats=args.out_feats, 
        num_layers=args.num_layers, 
        num_heads=args.num_heads, 
        feat_drop=args.feat_drop, 
        attn_drop=args.attn_drop, 
        do_train=True
    )

    pred = MLPPredictor(args.input_size, args.hidden_size)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')

    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)

    # Initialize StepLR scheduler
    ##scheduler = StepLR(optimizer, step_size=200, gamma=0.1)  # Adjust step_size and gamma as needed
    scheduler = ExponentialLR(optimizer, gamma=0.9) 

    output_path = './prediction/results/'
    os.makedirs(output_path, exist_ok=True)
    
    train_f1_scores = []
    val_f1_scores = []
    train_focal_loss_scores = []
    val_focal_loss_scores = []
    train_auc_scores = []
    val_auc_scores = []
    train_map_scores = []
    val_map_scores = []
    train_recall_scores = []
    val_recall_scores = []
    train_acc_scores = []
    val_acc_scores = []
    train_precision_scores = []
    val_precision_scores = []

    ##for epoch in range(num_epochs):
    for e in range(args.epochs):
        model.train()
        h = model(train_g, train_g.ndata['feat'])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)

        pos_labels = torch.ones_like(pos_score)
        neg_labels = torch.zeros_like(neg_score)

        all_scores = torch.cat([pos_score, neg_score])
        all_labels = torch.cat([pos_labels, neg_labels])

        loss = criterion(all_scores, all_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
     
        # Update the learning rate
        '''        
        scheduler.step()
 
        # Print the current learning rate
        if e % 200 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch {e}: Learning Rate = {current_lr:.6f}') 
        
        '''    
               
        if e % 5 == 0:
            print(f'In epoch {e}, loss: {loss.item()}')


        with torch.no_grad():
            h_train = model(train_g, train_g.ndata['feat'])
            train_pos_score = pred(train_pos_g, h_train)
            train_neg_score = pred(train_neg_g, h_train)
            train_f1 = compute_f1(train_pos_score, train_neg_score)
            train_f1_scores.append(train_f1.item())
            train_focal_loss= compute_focalloss(train_pos_score, train_neg_score)
            train_focal_loss_scores.append(train_focal_loss)
            train_auc = compute_auc(train_pos_score, train_neg_score)
            train_auc_scores.append(train_auc.item())
            train_map = compute_map(train_pos_score, train_neg_score)
            train_map_scores.append(train_map.item())
            train_recall = compute_recall(train_pos_score, train_neg_score)
            train_recall_scores.append(train_recall.item())
            train_acc = compute_accuracy(train_pos_score, train_neg_score)
            train_acc_scores.append(train_acc)
            train_precision = compute_precision(train_pos_score, train_neg_score)
            train_precision_scores.append(train_precision)

            h_val = model(train_g, train_g.ndata['feat'])
            val_pos_score = pred(val_pos_g, h_val)
            val_neg_score = pred(val_neg_g, h_val)
            val_f1 = compute_f1(val_pos_score, val_neg_score)
            val_f1_scores.append(val_f1.item())
            val_focal_loss= compute_focalloss(val_pos_score, val_neg_score)
            val_focal_loss_scores.append(val_focal_loss)
            val_auc = compute_auc(val_pos_score, val_neg_score)
            val_auc_scores.append(val_auc.item())
            val_map = compute_map(val_pos_score, val_neg_score)
            val_map_scores.append(val_map.item())
            val_recall = compute_recall(val_pos_score, val_neg_score)
            val_recall_scores.append(val_recall.item())
            val_acc = compute_accuracy(val_pos_score, val_neg_score)
            val_acc_scores.append(val_acc)
            val_precision = compute_precision(val_pos_score, val_neg_score)
            val_precision_scores.append(val_precision)

    epochs = range(args.epochs)
    ##epochs = list(map(int, epochs))

    
    with torch.no_grad():
        model.eval()
        h_test = model(G_dgl, G_dgl.ndata['feat'])
        test_pos_score = pred(test_pos_g, h_test)
        test_neg_score = pred(test_neg_g, h_test)
        test_auc, test_auc_err = compute_auc_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_f1, test_f1_err = compute_f1_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_focal_loss, test_focal_loss_err = compute_focalloss_with_symmetrical_confidence(test_pos_score, test_neg_score)#train_focal_loss, train_focal_loss_err
        test_precision, test_precision_err = compute_precision_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_recall, test_recall_err = compute_recall_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_hits_k = compute_hits_k(test_pos_score, test_neg_score, k=10)
        test_map, test_map_err = compute_map_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_accuracy, test_accuracy_err = compute_accuracy_with_symmetrical_confidence(test_pos_score, test_neg_score)

        print(f'Test AUC: {test_auc:.4f} ± {test_auc_err:.4f} | Test F1: {test_f1:.4f} ± {test_f1_err:.4f} | Test FocalLoss: {test_focal_loss:.4f} ± {test_focal_loss_err:.4f} |Test Accuracy: {test_accuracy:.4f} ± {test_accuracy_err:.4f} | Test Precision: {test_precision:.4f} ± {test_precision_err:.4f} | Test Recall: {test_recall:.4f} ± {test_recall_err:.4f} | Test mAP: {test_map:.4f} ± {test_map_err:.4f}')

    model_path = './prediction/results/pred_model.pth'
    torch.save(pred.state_dict(), model_path)
    


    test_auc = test_auc.item()
    test_f1 = test_f1.item()
    ##test_focal_loss = test_focal_loss.item()
    test_precision = test_precision.item()
    test_recall = test_recall.item()
    test_hits_k = test_hits_k.item()
    test_map = test_map.item()
    ##test_accuracy = test_accuracy.item()

    test_auc_err = test_auc_err.item()
    test_f1_err = test_f1_err.item()
    ##test_focal_loss_err = test_focal_loss_err.item()
    test_precision_err = test_precision_err.item()
    test_recall_err = test_recall_err.item()
    test_map_err = test_map_err.item()

    output = {
        'Test AUC': f'{test_auc:.4f} ± {test_auc_err:.4f}',
        'Test F1 Score': f'{test_f1:.4f} ± {test_f1_err:.4f}',
        'Test FocalLoss Score': f'{test_focal_loss:.4f} ± {test_focal_loss_err:.4f}',
        'Test Precision': f'{test_precision:.4f} ± {test_precision_err:.4f}',
        'Test Recall': f'{test_recall:.4f} ± {test_recall_err:.4f}',
        'Test Hit': f'{test_hits_k:.4f}',  # Assuming no confidence interval for Hits@K
        'Test mAP': f'{test_map:.4f} ± {test_map_err:.4f}',
        'Test Accuracy': f'{test_accuracy:.4f} ± {test_accuracy_err:.4f}'
    }

    filename_ = f'test_head{args.num_heads}_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.json'
    
    with open(os.path.join(output_path, filename_), 'w') as f:
        json.dump(output, f)

    filename = f'test_head{args.num_heads}_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.json'
    
    test_results = {
        'Learning Rate': args.lr,
        'Epochs': args.epochs,
        'Input Features': args.input_size,
        'Output Features': args.out_feats,
        'Test AUC': f'{test_auc:.4f} ± {test_auc_err:.4f}',
        'Test F1 Score': f'{test_f1:.4f} ± {test_f1_err:.4f}',
        'Test FocalLoss Score': f'{test_focal_loss:.4f} ± {test_focal_loss_err:.4f}',
        'Test Precision': f'{test_precision:.4f} ± {test_precision_err:.4f}',
        'Test Recall': f'{test_recall:.4f} ± {test_recall_err:.4f}',
        'Test Hit': f'{test_hits_k:.4f}',
        'Test mAP': f'{test_map:.4f} ± {test_map_err:.4f}',
        'Test Accuracy': f'{test_accuracy:.4f} ± {test_accuracy_err:.4f}'
    }

    with open(os.path.join(output_path, filename), 'w') as f:
        json.dump(test_results, f)

    '''plot_scores(epochs, train_f1_scores, val_f1_scores, train_focal_loss_scores, val_focal_loss_scores, train_auc_scores, val_auc_scores, 
        train_map_scores, val_map_scores, train_recall_scores, val_recall_scores,
        train_acc_scores, val_acc_scores, train_precision_scores, val_precision_scores,
        output_path, args)
    '''
    
def train_and_evaluate(args, G_dgl, node_features):
    # --- Edge splits (train/val/test) ---
    u, v = G_dgl.edges()
    eids = np.random.permutation(np.arange(G_dgl.number_of_edges()))
    test_size = int(len(eids) * 0.1)
    val_size = int(len(eids) * 0.1)
    train_eids = eids[test_size + val_size:]
    val_eids = eids[test_size:test_size + val_size]
    test_eids = eids[:test_size]

    def get_edge_split(eids_split):
        return u[eids_split], v[eids_split]

    train_pos_u, train_pos_v = get_edge_split(train_eids)
    val_pos_u, val_pos_v = get_edge_split(val_eids)
    test_pos_u, test_pos_v = get_edge_split(test_eids)

    # --- Negative edges ---
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())),
                        shape=(G_dgl.number_of_nodes(), G_dgl.number_of_nodes()))
    adj_neg = 1 - adj.todense() - np.eye(G_dgl.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    neg_eids = np.random.choice(len(neg_u), G_dgl.number_of_edges())
    train_neg_u, train_neg_v = neg_u[neg_eids[train_eids]], neg_v[neg_eids[train_eids]]
    val_neg_u, val_neg_v = neg_u[neg_eids[val_eids]], neg_v[neg_eids[val_eids]]
    test_neg_u, test_neg_v = neg_u[neg_eids[test_eids]], neg_v[neg_eids[test_eids]]

    def create_graph(u, v, num_nodes):
        return dgl.graph((u, v), num_nodes=num_nodes)

    train_pos_g = create_graph(train_pos_u, train_pos_v, G_dgl.number_of_nodes())
    train_neg_g = create_graph(train_neg_u, train_neg_v, G_dgl.number_of_nodes())
    val_pos_g = create_graph(val_pos_u, val_pos_v, G_dgl.number_of_nodes())
    val_neg_g = create_graph(val_neg_u, val_neg_v, G_dgl.number_of_nodes())
    test_pos_g = create_graph(test_pos_u, test_pos_v, G_dgl.number_of_nodes())
    test_neg_g = create_graph(test_neg_u, test_neg_v, G_dgl.number_of_nodes())

    # --- Model, predictor, loss, optimizer ---
    model = GATModel(node_features.shape[1], out_feats=args.out_feats,
                     num_layers=args.num_layers, num_heads=args.num_heads,
                     feat_drop=args.feat_drop, attn_drop=args.attn_drop, do_train=True)
    pred = MLPPredictor(args.input_size, args.hidden_size)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    output_path = './prediction/results/'
    os.makedirs(output_path, exist_ok=True)

    # --- Training loop ---
    for e in range(args.epochs):
        model.train()
        h = model(train_pos_g, train_pos_g.ndata['feat'])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        all_scores = torch.cat([pos_score, neg_score])
        all_labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        loss = criterion(all_scores, all_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if e % 5 == 0:
            print(f'Epoch {e}, Loss: {loss.item():.4f}')

    # --- Test predictions ---
    model.eval()
    with torch.no_grad():
        h_test = model(G_dgl, G_dgl.ndata['feat'])
        test_pos_score = pred(test_pos_g, h_test)

    # --- Louvain clustering on top 1000 predicted edges ---
    top_k = 1000
    scores = test_pos_score.detach().cpu().numpy()
    top_idx = np.argsort(-scores)[:top_k]
    u_test, v_test = G_dgl.edges()
    u_test, v_test = u_test.cpu().numpy(), v_test.cpu().numpy()
    top_u, top_v = u_test[top_idx], v_test[top_idx]

    subgraph = nx.Graph()
    for src, dst in zip(top_u, top_v):
        subgraph.add_edge(int(src), int(dst))

    partition = community.best_partition(subgraph)
    print(f"Louvain detected {len(set(partition.values()))} clusters")

    # --- Save cluster assignments and edges ---
    with open(os.path.join(output_path, 'topk_louvain_clusters.json'), 'w') as f:
        json.dump(partition, f)

    subgraph_edges = list(subgraph.edges())
    with open(os.path.join(output_path, 'topk_subgraph_edges.json'), 'w') as f:
        json.dump(subgraph_edges, f)

    # --- Visualization ---
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(subgraph, seed=42)
    cluster_ids = list(set(partition.values()))
    colors = plt.cm.get_cmap('tab20', len(cluster_ids))
    node_colors = [colors(partition[node]) for node in subgraph.nodes()]
    nx.draw_networkx(subgraph, pos, node_color=node_colors, with_labels=False, node_size=50, edge_color='gray')
    plt.title("Top 1000 Predicted Edges Louvain Clustering")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'topk_louvain_clusters.png'), dpi=300)
    plt.show()

    return partition, subgraph

import os
import json
import torch
import torch.nn.functional as F
import dgl
import numpy as np
import networkx as nx
import igraph as ig
import leidenalg as la
import matplotlib.pyplot as plt
import itertools

# ==========================
# Load JSON embeddings
# ==========================
def load_graph_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    node_ids = {}
    features = []
    for entry in data:
        for key in ['n', 'm']:
            node = entry[key]
            nid = node['identity']
            if nid not in node_ids:
                node_ids[nid] = len(node_ids)
                features.append(node['properties']['embedding'])
    
    node_features = torch.tensor(features, dtype=torch.float32)
    num_nodes = len(node_ids)
    
    src = []
    dst = []
    for entry in data:
        s = node_ids[entry['r']['start']]
        t = node_ids[entry['r']['end']]
        src.append(s)
        dst.append(t)
    
    G_dgl = dgl.graph((src, dst), num_nodes=num_nodes)
    G_dgl.ndata['feat'] = node_features
    
    return G_dgl, node_features, node_ids

# ==========================
# GAT + MLP model placeholders
# ==========================
class GATModel(torch.nn.Module):
    def __init__(self, in_feats, out_feats, num_layers=2, num_heads=1, feat_drop=0.0, attn_drop=0.0, do_train=True):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(dgl.nn.GATConv(in_feats, out_feats, num_heads, feat_drop, attn_drop, activation=F.elu))
        for _ in range(num_layers-1):
            self.layers.append(dgl.nn.GATConv(out_feats*num_heads, out_feats, num_heads, feat_drop, attn_drop, activation=F.elu))
        self.do_train = do_train

    def forward(self, g, h):
        for l in self.layers:
            h = l(g, h).flatten(1)
        return h

class MLPPredictor(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_feats*2, hidden_feats)
        self.fc2 = torch.nn.Linear(hidden_feats, 1)

    def forward(self, g, h):
        src, dst = g.edges()
        h_cat = torch.cat([h[src], h[dst]], dim=1)
        return torch.sigmoid(self.fc2(F.relu(self.fc1(h_cat)))).squeeze()

# ==========================
# Training function
# ==========================
def train_and_evaluate(args, G_dgl, node_features, top_k=1000):
    # Split edges
    u, v = G_dgl.edges()
    eids = np.arange(G_dgl.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    val_size = int(len(eids) * 0.1)

    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    val_pos_u, val_pos_v = u[eids[test_size:test_size+val_size]], v[eids[test_size:test_size+val_size]]
    train_pos_u, train_pos_v = u[eids[test_size+val_size:]], v[eids[test_size+val_size:]]

    adj = torch.zeros(G_dgl.number_of_nodes(), G_dgl.number_of_nodes())
    adj[u, v] = 1
    neg_u, neg_v = torch.where(adj == 0)
    neg_indices = np.random.choice(len(neg_u), len(u), replace=False)
    test_neg_u, test_neg_v = neg_u[neg_indices[:test_size]], neg_v[neg_indices[:test_size]]
    val_neg_u, val_neg_v = neg_u[neg_indices[test_size:test_size+val_size]], neg_v[neg_indices[test_size:test_size+val_size]]
    train_neg_u, train_neg_v = neg_u[neg_indices[test_size+val_size:]], neg_v[neg_indices[test_size+val_size:]]

    def create_graph(u, v):
        g = dgl.graph((u, v), num_nodes=G_dgl.number_of_nodes())
        g.ndata['feat'] = node_features
        return g

    train_pos_g = create_graph(train_pos_u, train_pos_v)
    train_neg_g = create_graph(train_neg_u, train_neg_v)
    val_pos_g = create_graph(val_pos_u, val_pos_v)
    val_neg_g = create_graph(val_neg_u, val_neg_v)
    test_pos_g = create_graph(test_pos_u, test_pos_v)
    test_neg_g = create_graph(test_neg_u, test_neg_v)

    # Model
    model = GATModel(node_features.shape[1], args.out_feats, args.num_layers, args.num_heads, args.feat_drop, args.attn_drop)
    pred = MLPPredictor(args.input_size, args.hidden_size)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)
    criterion = torch.nn.BCELoss()

    # Training loop
    for e in range(args.epochs):
        model.train()
        h = model(train_pos_g, train_pos_g.ndata['feat'])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        all_scores = torch.cat([pos_score, neg_score])
        all_labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        loss = criterion(all_scores, all_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if e % 20 == 0:
            print(f"Epoch {e}: Loss={loss.item():.4f}")

    # Top-k predictions
    h = model(G_dgl, node_features)
    all_edges = [(i,j) for i in range(G_dgl.number_of_nodes()) for j in range(G_dgl.number_of_nodes()) if i!=j]
    scores = []
    batch_size = 10000
    for i in range(0, len(all_edges), batch_size):
        batch = all_edges[i:i+batch_size]
        g_batch = dgl.graph(([u for u,v in batch], [v for u,v in batch]), num_nodes=G_dgl.number_of_nodes())
        g_batch.ndata['feat'] = node_features
        scores.extend(pred(g_batch, h).detach().cpu().numpy())
    top_idx = np.argsort(scores)[-top_k:]
    top_edges = [all_edges[i] for i in top_idx]
    return top_edges, model, pred

# ==========================
# Leiden clustering + visualization
# ==========================
def leiden_clustering(top_edges, node_ids):
    G_nx = nx.Graph()
    id_to_node = {v:k for k,v in node_ids.items()}
    for u,v in top_edges:
        G_nx.add_edge(u,v)
    G_ig = ig.Graph.from_networkx(G_nx)
    partition = la.find_partition(G_ig, la.ModularityVertexPartition)
    clusters = {id_to_node[i]: membership for i, membership in enumerate(partition.membership)}
    return clusters

def plot_clusters(top_edges, clusters, node_ids):
    G_nx = nx.Graph()
    for u,v in top_edges:
        G_nx.add_edge(u,v)
    colors = [clusters[u] for u in G_nx.nodes()]
    pos = nx.spring_layout(G_nx, seed=42)
    plt.figure(figsize=(12,12))
    nx.draw_networkx_nodes(G_nx, pos, node_color=colors, cmap='tab20', node_size=100)
    nx.draw_networkx_edges(G_nx, pos, alpha=0.3)
    plt.axis('off')
    plt.show()

import torch
import dgl
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from src.utils import MLPPredictor, GATModel, FocalLoss

def train_and_evaluate(args, G_dgl, node_features, top_k=1000):
    u, v = G_dgl.edges()
    eids = np.arange(G_dgl.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    val_size = int(len(eids) * 0.1)

    train_pos_u, train_pos_v = u[eids[test_size+val_size:]], v[eids[test_size+val_size:]]
    val_pos_u, val_pos_v = u[eids[test_size:test_size + val_size]], v[eids[test_size:test_size + val_size]]
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]

    def create_graph(u, v, num_nodes):
        return dgl.graph((u, v), num_nodes=num_nodes)

    train_g = create_graph(train_pos_u, train_pos_v, G_dgl.number_of_nodes())

    # Model setup
    model = GATModel(node_features.shape[1], out_feats=args.out_feats,
                     num_layers=args.num_layers, num_heads=args.num_heads,
                     feat_drop=args.feat_drop, attn_drop=args.attn_drop, do_train=True)
    pred = MLPPredictor(args.input_size, args.hidden_size)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    optimizer = torch.optim.Adam(list(model.parameters()) + list(pred.parameters()), lr=args.lr)

    # Training loop
    for e in range(args.epochs):
        model.train()
        h = model(train_g, train_g.ndata['feat'])
        # Simple full positive scores for demonstration
        pos_score = pred(train_g, h)
        pos_labels = torch.ones_like(pos_score)
        loss = criterion(pos_score, pos_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if e % 10 == 0:
            print(f"Epoch {e}, Loss: {loss.item():.4f}")

    # Predict all edges scores
    model.eval()
    with torch.no_grad():
        h_test = model(G_dgl, node_features)
        all_scores = pred(G_dgl, h_test)

    # Select top-k predictions
    topk_indices = torch.topk(all_scores.flatten(), top_k).indices
    edges = list(zip(u[topk_indices].tolist(), v[topk_indices].tolist()))
    return edges

# ==========================
# Main
# ==========================
'''if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-feats', type=int, default=128)
    parser.add_argument('--out-feats', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=1)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--input-size', type=int, default=2)
    parser.add_argument('--hidden-size', type=int, default=16)
    parser.add_argument('--feat-drop', type=float, default=0.0)
    parser.add_argument('--attn-drop', type=float, default=0.0)
    args = parser.parse_args()

    json_file = 'embedding/results/node_embeddings/neo4j_triplets_head1_dim128_lay2_epo20.json'
    G_dgl, node_features, node_ids = load_graph_from_json(json_file)
    top_edges, model, pred = train_and_evaluate(args, G_dgl, node_features, top_k=1000)
    clusters = leiden_clustering(top_edges, node_ids)
    plot_clusters(top_edges, clusters, node_ids)
'''

import os
import torch
import numpy as np
import dgl
import networkx as nx
import matplotlib.pyplot as plt
from src.models import GATModel, MLPPredictor, FocalLoss  # make sure these are defined/imported
from src.utils import compute_f1  # or other metrics if needed

def train_and_evaluate(args, G_dgl, node_features, top_k=1000, visualize=True):
    u, v = G_dgl.edges()
    eids = np.arange(G_dgl.number_of_edges())
    np.random.shuffle(eids)
    
    test_size = int(len(eids) * 0.1)
    val_size = int(len(eids) * 0.1)
    
    train_pos_u, train_pos_v = u[eids[test_size+val_size:]], v[eids[test_size+val_size:]]
    val_pos_u, val_pos_v = u[eids[test_size:test_size+val_size]], v[eids[test_size:test_size+val_size]]
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]

    # === Subgraph creation with node features
    def create_graph(u, v, num_nodes, features):
        g = dgl.graph((u, v), num_nodes=num_nodes)
        g.ndata['feat'] = features
        return g

    train_g = create_graph(train_pos_u, train_pos_v, G_dgl.number_of_nodes(), node_features)
    
    # Model
    model = GATModel(node_features.shape[1], out_feats=args.out_feats,
                     num_layers=args.num_layers, num_heads=args.num_heads,
                     feat_drop=args.feat_drop, attn_drop=args.attn_drop, do_train=True)
    pred = MLPPredictor(args.input_size, args.hidden_size)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    optimizer = torch.optim.Adam(list(model.parameters()) + list(pred.parameters()), lr=args.lr)
    
    # === Training loop
    for e in range(args.epochs):
        model.train()
        h = model(train_g, train_g.ndata['feat'])
        pos_score = pred(train_g, h)
        pos_labels = torch.ones_like(pos_score)
        loss = criterion(pos_score, pos_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if e % 10 == 0:
            print(f"Epoch {e}, Loss: {loss.item():.4f}")
    
    # === Top-k edge prediction
    model.eval()
    with torch.no_grad():
        h_test = model(G_dgl, node_features)
        all_scores = pred(G_dgl, h_test).flatten()
        topk_indices = torch.topk(all_scores, top_k).indices
        top_edges = list(zip(u[topk_indices].tolist(), v[topk_indices].tolist()))
    
    # === Visualization
    if visualize:
        nx_graph = dgl.to_networkx(G_dgl)
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(nx_graph, seed=42)
        nx.draw_networkx_nodes(nx_graph, pos, node_size=50, node_color='skyblue')
        nx.draw_networkx_edges(nx_graph, pos, alpha=0.2)
        # Highlight top-k predicted edges in red
        nx.draw_networkx_edges(nx_graph, pos, edgelist=top_edges, edge_color='red', width=2)
        plt.title(f"Top-{top_k} Predicted Edges")
        plt.show()
    
    return top_edges
