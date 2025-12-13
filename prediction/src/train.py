# ==============================
# Standard Library
# ==============================
import os
import json
import itertools

# ==============================
# Scientific / ML Libraries
# ==============================
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from scipy.stats import sem

# ==============================
# Graph Libraries
# ==============================
import dgl
from dgl.dataloading import GraphDataLoader
import networkx as nx

# ==============================
# Visualization
# ==============================
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ==============================
# Models & Utilities (local imports)
# ==============================
from .models import LinkPredictor, GATModel, ECGNN, MLPPredictor, FocalLoss
from .utils import (
    plot_scores, compute_hits_k, compute_auc, compute_f1, compute_focalloss,
    compute_accuracy, compute_precision, compute_recall, compute_map,
    compute_focalloss_with_symmetrical_confidence, compute_auc_with_symmetrical_confidence,
    compute_f1_with_symmetrical_confidence, compute_accuracy_with_symmetrical_confidence,
    compute_precision_with_symmetrical_confidence, compute_recall_with_symmetrical_confidence,
    compute_map_with_symmetrical_confidence
)

# ==============================
# Attribution / Clustering
# ==============================
from captum.attr import IntegratedGradients
from sklearn.cluster import SpectralBiclustering
from dgl.nn.pytorch.explain import GNNExplainer

def train_and_evaluate_(args, G_dgl, node_features):
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

    def create_graph(u, v, num_nodes):
        assert len(u) == len(v), "Source and destination nodes must have the same length"
        return dgl.graph((u, v), num_nodes=num_nodes)

    train_pos_g = create_graph(train_pos_u, train_pos_v, G_dgl.number_of_nodes())
    train_neg_g = create_graph(train_neg_u, train_neg_v, G_dgl.number_of_nodes())
    val_pos_g = create_graph(val_pos_u, val_pos_v, G_dgl.number_of_nodes())
    val_neg_g = create_graph(val_neg_u, val_neg_v, G_dgl.number_of_nodes())
    test_pos_g = create_graph(test_pos_u, test_pos_v, G_dgl.number_of_nodes())
    test_neg_g = create_graph(test_neg_u, test_neg_v, G_dgl.number_of_nodes())

    # ✅ FIX: pull hidden_feats & out_feats safely
    hidden_feats = getattr(args, "hidden_feats", 64)   # default 64 if not in args
    out_feats = getattr(args, "out_feats", 2)          # default 2 if not in args


    model = GATModel(
        # in_feats=in_feats,
        node_features.shape[1],
        hidden_feats=hidden_feats,
        out_feats=out_feats,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        feat_drop=0.2,
        attn_drop=0.2,
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

def ig_forward(model, G_dgl, feat_tensor):
    """
    Forward wrapper for Integrated Gradients attribution.
    Ensures the feature tensor matches the number of nodes in G_dgl.
    """
    num_nodes = G_dgl.number_of_nodes()
    if feat_tensor.shape[0] != num_nodes:
        feat_tensor = feat_tensor[:num_nodes]
        if len(feat_tensor.shape) == 1:
            feat_tensor = feat_tensor.view(num_nodes, 1)
    G_dgl.ndata['feat'] = feat_tensor
    return model(G_dgl, feat_tensor)

def train_and_evaluate__(args, G_dgl, node_features):
    # ---- Split edges ----
    u, v = G_dgl.edges()
    eids = np.arange(G_dgl.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    val_size = int(len(eids) * 0.1)

    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    val_pos_u, val_pos_v = u[eids[test_size:test_size + val_size]], v[eids[test_size:test_size + val_size]]
    train_pos_u, train_pos_v = u[eids[test_size + val_size:]], v[eids[test_size + val_size:]]

    adj = sp.coo_matrix(
        (np.ones(len(u)), (u.numpy(), v.numpy())),
        shape=(G_dgl.number_of_nodes(), G_dgl.number_of_nodes())
    )
    adj_neg = 1 - adj.todense() - np.eye(G_dgl.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    neg_eids = np.random.choice(len(neg_u), G_dgl.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    val_neg_u, val_neg_v = neg_u[neg_eids[test_size:test_size + val_size]], neg_v[neg_eids[test_size:test_size + val_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size + val_size:]], neg_v[neg_eids[test_size + val_size:]]

    train_g = dgl.remove_edges(G_dgl, eids[:test_size + val_size])

    def create_graph(u, v, num_nodes):
        return dgl.graph((u, v), num_nodes=num_nodes)

    train_pos_g = create_graph(train_pos_u, train_pos_v, G_dgl.number_of_nodes())
    train_neg_g = create_graph(train_neg_u, train_neg_v, G_dgl.number_of_nodes())
    val_pos_g = create_graph(val_pos_u, val_pos_v, G_dgl.number_of_nodes())
    val_neg_g = create_graph(val_neg_u, val_neg_v, G_dgl.number_of_nodes())
    test_pos_g = create_graph(test_pos_u, test_pos_v, G_dgl.number_of_nodes())
    test_neg_g = create_graph(test_neg_u, test_neg_v, G_dgl.number_of_nodes())

    # ✅ FIX: pull hidden_feats & out_feats safely
    hidden_feats = getattr(args, "hidden_feats", 64)   # default 64 if not in args
    out_feats = getattr(args, "out_feats", 2)          # default 2 if not in args


    model = GATModel(
        # in_feats=in_feats,
        node_features.shape[1],
        hidden_feats=hidden_feats,
        out_feats=out_feats,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        feat_drop=0.2,
        attn_drop=0.2,
    )
    pred = MLPPredictor(args.input_size, args.hidden_size)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    output_path = './prediction/results/'
    os.makedirs(output_path, exist_ok=True)

    # ---- Training Loop ----
    for e in range(args.epochs):
        model.train()
        h = model(train_g, train_g.ndata['feat'])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)

        all_scores = torch.cat([pos_score, neg_score])
        all_labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        loss = criterion(all_scores, all_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if e % 5 == 0:
            print(f'Epoch {e}, loss: {loss.item():.4f}')

    # ---- Evaluation ----
    with torch.no_grad():
        model.eval()
        h_test = model(G_dgl, G_dgl.ndata['feat'])
        test_pos_score = pred(test_pos_g, h_test)
        test_neg_score = pred(test_neg_g, h_test)

        # Metrics with confidence
        test_auc, test_auc_err = compute_auc_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_f1, test_f1_err = compute_f1_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_focal_loss, test_focal_loss_err = compute_focalloss_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_precision, test_precision_err = compute_precision_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_recall, test_recall_err = compute_recall_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_map, test_map_err = compute_map_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_accuracy, test_accuracy_err = compute_accuracy_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_hits_k = compute_hits_k(test_pos_score, test_neg_score, k=10)

        print(f'Test AUC: {test_auc:.4f} ± {test_auc_err:.4f} | '
              f'Test F1: {test_f1:.4f} ± {test_f1_err:.4f} | '
              f'Test FocalLoss: {test_focal_loss:.4f} ± {test_focal_loss_err:.4f}')

    # ---- Save model ----
    model_path = os.path.join(output_path, 'pred_model.pth')
    torch.save(pred.state_dict(), model_path)

    # ---- Save results ----
    results = {
        'Test AUC': f'{test_auc:.4f} ± {test_auc_err:.4f}',
        'Test F1 Score': f'{test_f1:.4f} ± {test_f1_err:.4f}',
        'Test FocalLoss Score': f'{test_focal_loss:.4f} ± {test_focal_loss_err:.4f}',
        'Test Precision': f'{test_precision:.4f} ± {test_precision_err:.4f}',
        'Test Recall': f'{test_recall:.4f} ± {test_recall_err:.4f}',
        'Test Hit': f'{test_hits_k:.4f}',
        'Test mAP': f'{test_map:.4f} ± {test_map_err:.4f}',
        'Test Accuracy': f'{test_accuracy:.4f} ± {test_accuracy_err:.4f}'
    }

    filename = f'test_head{args.num_heads}_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.json'
    with open(os.path.join(output_path, filename), 'w') as f:
        json.dump(results, f)

    # ---- Optional: Integrated Gradients attribution ----
    ig = IntegratedGradients(lambda feat: ig_forward(model, G_dgl, feat))
    node_features_ig = G_dgl.ndata['feat'].clone().detach()
    node_attributions = ig.attribute(node_features_ig, n_steps=50).detach().cpu().numpy()

    return results, node_attributions

def plot_epoch_metrics(epoch_metrics, output_path, args):
    epochs = range(1, len(epoch_metrics['train']['F1']) + 1)

    # Metrics to plot
    metrics = ['F1', 'AUC', 'Precision', 'Recall', 'FocalLoss']

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, epoch_metrics['train'][metric], label=f'Train {metric}', marker='o')
        plt.plot(epochs, epoch_metrics['val'][metric], label=f'Val {metric}', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'{metric} over Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save figure
        filename = f'{metric}_epoch_plot_head{args.num_heads}_lr{args.lr}_lay{args.num_layers}.png'
        plt.savefig(os.path.join(output_path, filename))
        plt.close()

def train_and_evaluate___(args, G_dgl, node_features):
    # ---- Split edges ----
    u, v = G_dgl.edges()
    eids = np.arange(G_dgl.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    val_size = int(len(eids) * 0.1)

    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    val_pos_u, val_pos_v = u[eids[test_size:test_size + val_size]], v[eids[test_size:test_size + val_size]]
    train_pos_u, train_pos_v = u[eids[test_size + val_size:]], v[eids[test_size + val_size:]]

    adj = sp.coo_matrix(
        (np.ones(len(u)), (u.numpy(), v.numpy())),
        shape=(G_dgl.number_of_nodes(), G_dgl.number_of_nodes())
    )
    adj_neg = 1 - adj.todense() - np.eye(G_dgl.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    neg_eids = np.random.choice(len(neg_u), G_dgl.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    val_neg_u, val_neg_v = neg_u[neg_eids[test_size:test_size + val_size]], neg_v[neg_eids[test_size:test_size + val_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size + val_size:]], neg_v[neg_eids[test_size + val_size:]]

    train_g = dgl.remove_edges(G_dgl, eids[:test_size + val_size])

    def create_graph(u, v, num_nodes):
        return dgl.graph((u, v), num_nodes=num_nodes)

    train_pos_g = create_graph(train_pos_u, train_pos_v, G_dgl.number_of_nodes())
    train_neg_g = create_graph(train_neg_u, train_neg_v, G_dgl.number_of_nodes())
    val_pos_g = create_graph(val_pos_u, val_pos_v, G_dgl.number_of_nodes())
    val_neg_g = create_graph(val_neg_u, val_neg_v, G_dgl.number_of_nodes())
    test_pos_g = create_graph(test_pos_u, test_pos_v, G_dgl.number_of_nodes())
    test_neg_g = create_graph(test_neg_u, test_neg_v, G_dgl.number_of_nodes())

    # ✅ FIX: pull hidden_feats & out_feats safely
    hidden_feats = getattr(args, "hidden_feats", 64)   # default 64 if not in args
    out_feats = getattr(args, "out_feats", 2)          # default 2 if not in args


    model = GATModel(
        # in_feats=in_feats,
        node_features.shape[1],
        hidden_feats=hidden_feats,
        out_feats=out_feats,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        feat_drop=0.2,
        attn_drop=0.2,
    )
    pred = MLPPredictor(args.input_size, args.hidden_size)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    output_path = './prediction/results/'
    os.makedirs(output_path, exist_ok=True)

    # ---- Initialize per-epoch metrics ----
    epoch_metrics = {
        'train': {'F1': [], 'AUC': [], 'Precision': [], 'Recall': [], 'FocalLoss': []},
        'val': {'F1': [], 'AUC': [], 'Precision': [], 'Recall': [], 'FocalLoss': []}
    }

    # ---- Training Loop ----
    for e in range(args.epochs):
        model.train()
        h = model(train_g, train_g.ndata['feat'])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)

        all_scores = torch.cat([pos_score, neg_score])
        all_labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        loss = criterion(all_scores, all_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if e % 5 == 0:
            print(f'Epoch {e}, loss: {loss.item():.4f}')

        # ---- Compute per-epoch train/val metrics ----
        with torch.no_grad():
            h_train = model(train_g, train_g.ndata['feat'])
            train_pos_score = pred(train_pos_g, h_train)
            train_neg_score = pred(train_neg_g, h_train)

            epoch_metrics['train']['F1'].append(compute_f1(train_pos_score, train_neg_score).item())
            epoch_metrics['train']['AUC'].append(compute_auc(train_pos_score, train_neg_score).item())
            epoch_metrics['train']['Precision'].append(compute_precision(train_pos_score, train_neg_score))
            epoch_metrics['train']['Recall'].append(compute_recall(train_pos_score, train_neg_score).item())
            epoch_metrics['train']['FocalLoss'].append(compute_focalloss(train_pos_score, train_neg_score))

            h_val = model(train_g, train_g.ndata['feat'])
            val_pos_score = pred(val_pos_g, h_val)
            val_neg_score = pred(val_neg_g, h_val)

            epoch_metrics['val']['F1'].append(compute_f1(val_pos_score, val_neg_score).item())
            epoch_metrics['val']['AUC'].append(compute_auc(val_pos_score, val_neg_score).item())
            epoch_metrics['val']['Precision'].append(compute_precision(val_pos_score, val_neg_score))
            epoch_metrics['val']['Recall'].append(compute_recall(val_pos_score, val_neg_score).item())
            epoch_metrics['val']['FocalLoss'].append(compute_focalloss(val_pos_score, val_neg_score))

    # ---- Evaluation ----
    with torch.no_grad():
        model.eval()
        h_test = model(G_dgl, G_dgl.ndata['feat'])
        test_pos_score = pred(test_pos_g, h_test)
        test_neg_score = pred(test_neg_g, h_test)

        test_auc, test_auc_err = compute_auc_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_f1, test_f1_err = compute_f1_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_focal_loss, test_focal_loss_err = compute_focalloss_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_precision, test_precision_err = compute_precision_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_recall, test_recall_err = compute_recall_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_map, test_map_err = compute_map_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_accuracy, test_accuracy_err = compute_accuracy_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_hits_k = compute_hits_k(test_pos_score, test_neg_score, k=10)

    # ---- Save model ----
    torch.save(pred.state_dict(), os.path.join(output_path, 'pred_model.pth'))

    # ---- Save final test results ----
    test_results = {
        'Test AUC': f'{test_auc:.4f} ± {test_auc_err:.4f}',
        'Test F1 Score': f'{test_f1:.4f} ± {test_f1_err:.4f}',
        'Test FocalLoss Score': f'{test_focal_loss:.4f} ± {test_focal_loss_err:.4f}',
        'Test Precision': f'{test_precision:.4f} ± {test_precision_err:.4f}',
        'Test Recall': f'{test_recall:.4f} ± {test_recall_err:.4f}',
        'Test Hit': f'{test_hits_k:.4f}',
        'Test mAP': f'{test_map:.4f} ± {test_map_err:.4f}',
        'Test Accuracy': f'{test_accuracy:.4f} ± {test_accuracy_err:.4f}',
        'Epoch Metrics': epoch_metrics  # Save all per-epoch metrics
    }

    filename = f'test_head{args.num_heads}_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.json'
    with open(os.path.join(output_path, filename), 'w') as f:
        json.dump(test_results, f, indent=4)

    # ---- Integrated Gradients Attribution ----
    ig = IntegratedGradients(lambda feat: ig_forward(model, G_dgl, feat))
    node_features_ig = G_dgl.ndata['feat'].clone().detach()
    node_attributions = ig.attribute(node_features_ig, n_steps=50).detach().cpu().numpy()

    return test_results, node_attributions

def train_and_evaluate_(args, G_dgl, node_features):
    # ---- Split edges ----
    u, v = G_dgl.edges()
    eids = np.arange(G_dgl.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    val_size = int(len(eids) * 0.1)

    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    val_pos_u, val_pos_v = u[eids[test_size:test_size + val_size]], v[eids[test_size:test_size + val_size]]
    train_pos_u, train_pos_v = u[eids[test_size + val_size:]], v[eids[test_size + val_size:]]

    adj = sp.coo_matrix(
        (np.ones(len(u)), (u.numpy(), v.numpy())),
        shape=(G_dgl.number_of_nodes(), G_dgl.number_of_nodes())
    )
    adj_neg = 1 - adj.todense() - np.eye(G_dgl.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    neg_eids = np.random.choice(len(neg_u), G_dgl.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    val_neg_u, val_neg_v = neg_u[neg_eids[test_size:test_size + val_size]], neg_v[neg_eids[test_size:test_size + val_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size + val_size:]], neg_v[neg_eids[test_size + val_size:]]

    train_g = dgl.remove_edges(G_dgl, eids[:test_size + val_size])

    def create_graph(u, v, num_nodes):
        return dgl.graph((u, v), num_nodes=num_nodes)

    train_pos_g = create_graph(train_pos_u, train_pos_v, G_dgl.number_of_nodes())
    train_neg_g = create_graph(train_neg_u, train_neg_v, G_dgl.number_of_nodes())
    val_pos_g = create_graph(val_pos_u, val_pos_v, G_dgl.number_of_nodes())
    val_neg_g = create_graph(val_neg_u, val_neg_v, G_dgl.number_of_nodes())
    test_pos_g = create_graph(test_pos_u, test_pos_v, G_dgl.number_of_nodes())
    test_neg_g = create_graph(test_neg_u, test_neg_v, G_dgl.number_of_nodes())

    # ✅ FIX: pull hidden_feats & out_feats safely
    hidden_feats = getattr(args, "hidden_feats", 64)   # default 64 if not in args
    out_feats = getattr(args, "out_feats", 2)          # default 2 if not in args


    model = GATModel(
        # in_feats=in_feats,
        node_features.shape[1],
        hidden_feats=hidden_feats,
        out_feats=out_feats,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        feat_drop=0.2,
        attn_drop=0.2,
    )
    pred = MLPPredictor(args.input_size, args.hidden_size)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    output_path = './prediction/results/'
    os.makedirs(output_path, exist_ok=True)

    # ---- Initialize per-epoch metrics ----
    epoch_metrics = {
        'train': {'F1': [], 'AUC': [], 'Precision': [], 'Recall': [], 'FocalLoss': []},
        'val': {'F1': [], 'AUC': [], 'Precision': [], 'Recall': [], 'FocalLoss': []}
    }

    # ---- Training Loop ----
    for e in range(args.epochs):
        model.train()
        h = model(train_g, train_g.ndata['feat'])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)

        all_scores = torch.cat([pos_score, neg_score])
        all_labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        loss = criterion(all_scores, all_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if e % 5 == 0:
            print(f'Epoch {e}, loss: {loss.item():.4f}')

        # ---- Compute per-epoch train/val metrics ----
        with torch.no_grad():
            h_train = model(train_g, train_g.ndata['feat'])
            train_pos_score = pred(train_pos_g, h_train)
            train_neg_score = pred(train_neg_g, h_train)

            epoch_metrics['train']['F1'].append(compute_f1(train_pos_score, train_neg_score).item())
            epoch_metrics['train']['AUC'].append(compute_auc(train_pos_score, train_neg_score).item())
            epoch_metrics['train']['Precision'].append(compute_precision(train_pos_score, train_neg_score))
            epoch_metrics['train']['Recall'].append(compute_recall(train_pos_score, train_neg_score).item())
            epoch_metrics['train']['FocalLoss'].append(compute_focalloss(train_pos_score, train_neg_score))

            h_val = model(train_g, train_g.ndata['feat'])
            val_pos_score = pred(val_pos_g, h_val)
            val_neg_score = pred(val_neg_g, h_val)

            epoch_metrics['val']['F1'].append(compute_f1(val_pos_score, val_neg_score).item())
            epoch_metrics['val']['AUC'].append(compute_auc(val_pos_score, val_neg_score).item())
            epoch_metrics['val']['Precision'].append(compute_precision(val_pos_score, val_neg_score))
            epoch_metrics['val']['Recall'].append(compute_recall(val_pos_score, val_neg_score).item())
            epoch_metrics['val']['FocalLoss'].append(compute_focalloss(val_pos_score, val_neg_score))

    # ---- Evaluation ----
    with torch.no_grad():
        model.eval()
        h_test = model(G_dgl, G_dgl.ndata['feat'])
        test_pos_score = pred(test_pos_g, h_test)
        test_neg_score = pred(test_neg_g, h_test)

        test_auc, test_auc_err = compute_auc_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_f1, test_f1_err = compute_f1_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_focal_loss, test_focal_loss_err = compute_focalloss_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_precision, test_precision_err = compute_precision_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_recall, test_recall_err = compute_recall_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_map, test_map_err = compute_map_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_accuracy, test_accuracy_err = compute_accuracy_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_hits_k = compute_hits_k(test_pos_score, test_neg_score, k=10)

    # ---- Save model ----
    torch.save(pred.state_dict(), os.path.join(output_path, 'pred_model.pth'))

    # ---- Save final test results ----
    test_results = {
        'Test AUC': f'{test_auc:.4f} ± {test_auc_err:.4f}',
        'Test F1 Score': f'{test_f1:.4f} ± {test_f1_err:.4f}',
        'Test FocalLoss Score': f'{test_focal_loss:.4f} ± {test_focal_loss_err:.4f}',
        'Test Precision': f'{test_precision:.4f} ± {test_precision_err:.4f}',
        'Test Recall': f'{test_recall:.4f} ± {test_recall_err:.4f}',
        'Test Hit': f'{test_hits_k:.4f}',
        'Test mAP': f'{test_map:.4f} ± {test_map_err:.4f}',
        'Test Accuracy': f'{test_accuracy:.4f} ± {test_accuracy_err:.4f}',
        'Epoch Metrics': epoch_metrics  # Save all per-epoch metrics
    }

    filename = f'test_head{args.num_heads}_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.json'
    with open(os.path.join(output_path, filename), 'w') as f:
        json.dump(test_results, f, indent=4)

    # ---- Integrated Gradients Attribution ----
    ig = IntegratedGradients(lambda feat: ig_forward(model, G_dgl, feat))
    node_features_ig = G_dgl.ndata['feat'].clone().detach()
    node_attributions = ig.attribute(node_features_ig, n_steps=50).detach().cpu().numpy()

    return test_results, node_attributions

# Assume these are already imported or defined elsewhere:
# GATModel, MLPPredictor, FocalLoss
# compute_f1, compute_auc, compute_precision, compute_recall, compute_map, compute_accuracy
# compute_focalloss
# compute_auc_with_symmetrical_confidence, compute_f1_with_symmetrical_confidence
# compute_focalloss_with_symmetrical_confidence, compute_precision_with_symmetrical_confidence
# compute_recall_with_symmetrical_confidence, compute_map_with_symmetrical_confidence
# compute_accuracy_with_symmetrical_confidence
# compute_hits_k

def plot_epoch_metrics(epoch_metrics, output_path, args):
    epochs = range(1, len(epoch_metrics['train']['F1']) + 1)
    metrics = ['F1', 'AUC', 'Precision', 'Recall', 'FocalLoss']

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, epoch_metrics['train'][metric], label=f'Train {metric}', marker='o')
        plt.plot(epochs, epoch_metrics['val'][metric], label=f'Val {metric}', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'{metric} over Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filename = f'{metric}_epoch_plot_head{args.num_heads}_lr{args.lr}_lay{args.num_layers}.png'
        plt.savefig(os.path.join(output_path, filename))
        plt.close()

def train_and_evaluate(args, G_dgl, node_features):
    u, v = G_dgl.edges()
    eids = np.arange(G_dgl.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    val_size = int(len(eids) * 0.1)

    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    val_pos_u, val_pos_v = u[eids[test_size:test_size + val_size]], v[eids[test_size:test_size + val_size]]
    train_pos_u, train_pos_v = u[eids[test_size + val_size:]], v[eids[test_size + val_size:]]

    adj = sp.coo_matrix(
        (np.ones(len(u)), (u.numpy(), v.numpy())),
        shape=(G_dgl.number_of_nodes(), G_dgl.number_of_nodes())
    )
    adj_neg = 1 - adj.todense() - np.eye(G_dgl.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), G_dgl.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    val_neg_u, val_neg_v = neg_u[neg_eids[test_size:test_size + val_size]], neg_v[neg_eids[test_size:test_size + val_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size + val_size:]], neg_v[neg_eids[test_size + val_size:]]

    train_g = dgl.remove_edges(G_dgl, eids[:test_size + val_size])

    def create_graph(u, v, num_nodes):
        assert len(u) == len(v), "Source and destination nodes must have the same length"
        return dgl.graph((u, v), num_nodes=num_nodes)

    train_pos_g = create_graph(train_pos_u, train_pos_v, G_dgl.number_of_nodes())
    train_neg_g = create_graph(train_neg_u, train_neg_v, G_dgl.number_of_nodes())
    val_pos_g = create_graph(val_pos_u, val_pos_v, G_dgl.number_of_nodes())
    val_neg_g = create_graph(val_neg_u, val_neg_v, G_dgl.number_of_nodes())
    test_pos_g = create_graph(test_pos_u, test_pos_v, G_dgl.number_of_nodes())
    test_neg_g = create_graph(test_neg_u, test_neg_v, G_dgl.number_of_nodes())

    # ✅ FIX: pull hidden_feats & out_feats safely
    hidden_feats = getattr(args, "hidden_feats", 64)   # default 64 if not in args
    out_feats = getattr(args, "out_feats", 2)          # default 2 if not in args

    in_feats = node_features.shape[1]
    model = ECGNN(in_feats, hidden_feats=64, out_feats=out_feats, k=3)
    # model = ECGNN(G_dgl, in_feats, hidden_feats, out_feats)

    
    pred = MLPPredictor(args.input_size, args.hidden_size)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    output_path = './prediction/results/'
    os.makedirs(output_path, exist_ok=True)

    # Initialize epoch metric tracking
    epoch_metrics = {
        'train': {'F1': [], 'AUC': [], 'Precision': [], 'Recall': [], 'FocalLoss': []},
        'val': {'F1': [], 'AUC': [], 'Precision': [], 'Recall': [], 'FocalLoss': []}
    }

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

        if e % 5 == 0:
            print(f'Epoch {e}, loss: {loss.item():.4f}')
            
        with torch.no_grad():
            # ---- Train Metrics ----
            h_train = model(train_g, train_g.ndata['feat'])
            train_pos_score = pred(train_pos_g, h_train)
            train_neg_score = pred(train_neg_g, h_train)
            epoch_metrics['train']['F1'].append(compute_f1(train_pos_score, train_neg_score).item())
            epoch_metrics['train']['AUC'].append(compute_auc(train_pos_score, train_neg_score).item())
            epoch_metrics['train']['Precision'].append(compute_precision(train_pos_score, train_neg_score))
            epoch_metrics['train']['Recall'].append(compute_recall(train_pos_score, train_neg_score))
            epoch_metrics['train']['FocalLoss'].append(compute_focalloss(train_pos_score, train_neg_score))

            # ---- Validation Metrics ----
            h_val = model(train_g, train_g.ndata['feat'])
            val_pos_score = pred(val_pos_g, h_val)
            val_neg_score = pred(val_neg_g, h_val)
            epoch_metrics['val']['F1'].append(compute_f1(val_pos_score, val_neg_score).item())
            epoch_metrics['val']['AUC'].append(compute_auc(val_pos_score, val_neg_score).item())
            epoch_metrics['val']['Precision'].append(compute_precision(val_pos_score, val_neg_score))
            epoch_metrics['val']['Recall'].append(compute_recall(val_pos_score, val_neg_score))
            epoch_metrics['val']['FocalLoss'].append(compute_focalloss(val_pos_score, val_neg_score))


    # ----------------------------
    # 5. Evaluate on test set
    # ----------------------------

    # ---- Test evaluation ----
    with torch.no_grad():
        model.eval()
        h_test = model(G_dgl, G_dgl.ndata['feat'])
        test_pos_score = pred(test_pos_g, h_test)
        test_neg_score = pred(test_neg_g, h_test)

        test_auc, test_auc_err = compute_auc_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_f1, test_f1_err = compute_f1_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_focal_loss, test_focal_loss_err = compute_focalloss_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_precision, test_precision_err = compute_precision_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_recall, test_recall_err = compute_recall_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_hits_k = compute_hits_k(test_pos_score, test_neg_score, k=10)
        test_map, test_map_err = compute_map_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_accuracy, test_accuracy_err = compute_accuracy_with_symmetrical_confidence(test_pos_score, test_neg_score)

        print(f'Test AUC: {test_auc:.4f} ± {test_auc_err:.4f} | Test F1: {test_f1:.4f} ± {test_f1_err:.4f}')

    # Save predictor model
    torch.save(pred.state_dict(), os.path.join(output_path, 'pred_model.pth'))

    # ---- Save JSON results ----
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

    filename = f'test_head{args.num_heads}_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.json'
    with open(os.path.join(output_path, filename), 'w') as f:
        json.dump(test_results, f)
        
    src, dst = int(test_pos_u[0]), int(test_pos_v[0])  # pick a test pair
    explanation = explain_pathway_link_dgl(
        model=model,
        G_dgl=G_dgl,
        x=G_dgl.ndata['feat'],   # pass node features
        src=src,
        dst=dst,
        node_names={i: f"Pathway_{i}" for i in range(G_dgl.num_nodes())},
        top_k=15
    )

    # the actual NetworkX subgraph:
    H = explanation["graph"]
    output_file = "prediction/results/reactome_link_explanation_node1935.png"
    plot_explanation(H, explanation, output_file=output_file)

    # ---- Plot per-epoch metrics ----
    plot_epoch_metrics(epoch_metrics, output_path, args)

    run_post_training_analysis(epoch_metrics, h_test, pred, G_dgl, output_path, args)

    # return test_results
    # ==============================
    # 🔎 Attribution + Visualization
    # ==============================
    print("⚡ Computing node importance with attributions ...")
    node_attributions = h_test.detach().cpu().numpy()   # shape (N, F)
    
    # Visualization functions
    plot_node_importance_graph(G_dgl, node_attributions, output_path)
    plot_node_feature_heatmap(node_attributions, output_path)
    plot_spectral_biclustering(node_attributions, output_path, topk=1000)
    
    # node_importance = aggregate_node_importance(node_attributions)

    # Save results and plots
    # save_node_importance(node_importance, output_path)

    # Plot attribution visualization
    # plot_graph_attributions(G_dgl, node_importance)
    # plot_feature_heatmap(node_attributions, output_path)

    # Print top-10 nodes
    # print_top_nodes(node_importance, G_dgl, topk=10)

    # # ==============================
    # # 💾 Save predictor model
    # # ==============================
    # torch.save(pred.state_dict(), os.path.join(output_path, 'pred_model.pth'))

    # Compute node attributions from test embeddings
    node_attributions = h_test.detach().cpu().numpy()  # shape: (num_nodes, feature_dim)

    # Compute importance per node (sum of absolute feature attributions)
    node_importance = np.abs(node_attributions).sum(axis=1)

    # Select top 1000 nodes
    topk = min(1000, len(node_importance))
    top_indices = np.argsort(node_importance)[::-1][:topk]
    top_node_attr = node_attributions[top_indices, :]

    # ---- Spectral Biclustering ----
    n_clusters_row = 4  # clusters for nodes
    n_clusters_col = 4  # clusters for features

    bicluster = SpectralBiclustering(n_clusters=(n_clusters_row, n_clusters_col), random_state=42)
    bicluster.fit(np.abs(top_node_attr))

    # Reorder according to bicluster labels
    fit_data = top_node_attr[np.argsort(bicluster.row_labels_)]
    fit_data = fit_data[:, np.argsort(bicluster.column_labels_)]

    # ---- Plot heatmap with cluster boundaries ----
    plt.figure(figsize=(12, 8))
    plt.imshow(fit_data, aspect='auto', cmap='RdBu_r')
    plt.colorbar(label="Attribution Value")
    plt.xlabel("Feature Clustered Index")
    plt.ylabel("Top 1000 Node Clustered Index")
    plt.title("Spectral Biclustering: Top 1000 Node-Feature Attribution Heatmap")

    # Add cluster boundaries
    row_splits = np.cumsum(np.bincount(bicluster.row_labels_)[np.argsort(np.unique(bicluster.row_labels_))])
    col_splits = np.cumsum(np.bincount(bicluster.column_labels_)[np.argsort(np.unique(bicluster.column_labels_))])

    for r in row_splits[:-1]:
        plt.axhline(r - 0.5, color='black', linewidth=1.2)
    for c in col_splits[:-1]:
        plt.axvline(c - 0.5, color='black', linewidth=1.2)

    plt.show()
    plt.savefig(os.path.join(output_path, "top1000_node_feature_bicluster_heatmap.png"), dpi=300)
    print(f"Saved Spectral Biclustering heatmap (top 1000 nodes) to {output_path}")

    # # ---- Spectral Biclustering ----
    # # node_attributions: shape (num_nodes, num_features)
    # n_clusters_row = 4  # number of clusters for nodes
    # n_clusters_col = 4  # number of clusters for features

    # bicluster = SpectralBiclustering(n_clusters=(n_clusters_row, n_clusters_col), random_state=42)
    # bicluster.fit(np.abs(node_attributions))  # absolute value to focus on magnitude

    # # Reorder rows and columns according to bicluster labels
    # fit_data = node_attributions[np.argsort(bicluster.row_labels_)]
    # fit_data = fit_data[:, np.argsort(bicluster.column_labels_)]

    # # ---- Plot heatmap with clusters ----
    # plt.figure(figsize=(12, 8))
    # plt.imshow(fit_data, aspect='auto', cmap='RdBu_r')
    # plt.colorbar(label="Attribution Value")
    # plt.xlabel("Feature Clustered Index")
    # plt.ylabel("Node Clustered Index")
    # plt.title("Spectral Biclustering: Node-Feature Attribution Heatmap")

    # # Add cluster boundaries
    # row_splits = np.cumsum(np.bincount(bicluster.row_labels_)[np.argsort(np.unique(bicluster.row_labels_))])
    # col_splits = np.cumsum(np.bincount(bicluster.column_labels_)[np.argsort(np.unique(bicluster.column_labels_))])

    # for r in row_splits[:-1]:
    #     plt.axhline(r - 0.5, color='black', linewidth=1.2)
    # for c in col_splits[:-1]:
    #     plt.axvline(c - 0.5, color='black', linewidth=1.2)

    # plt.show()
    # plt.savefig(os.path.join(output_path, "node_feature_bicluster_heatmap.png"), dpi=300)
    # print(f"Saved Spectral Biclustering heatmap to {output_path}")

    # # ==============================
    # # 7. Visualization of Attributions
    # # ==============================

    # Aggregate importance per node (sum across features)
    node_importance = np.abs(node_attributions).sum(axis=1)

    # Normalize for coloring
    node_importance_norm = (node_importance - node_importance.min()) / (node_importance.max() - node_importance.min() + 1e-9)

    # ---- Build NetworkX graph for plotting ----
    G_nx = G_dgl.to_networkx().to_undirected()

    # Position nodes (spring layout or graphviz if available)
    pos = nx.spring_layout(G_nx, seed=42)

    # Plot graph with node importance as color intensity
    plt.figure(figsize=(10, 8))
    nodes = nx.draw_networkx_nodes(
        G_nx, pos,
        node_size=100,
        node_color=node_importance_norm,
        cmap=plt.cm.viridis
    )
    nx.draw_networkx_edges(G_nx, pos, alpha=0.3)
    plt.colorbar(nodes, label="Node Attribution Importance")
    plt.title("Integrated Gradients: Node Feature Importance", fontsize=14)
    plt.axis("off")
    plt.show()

    # ---- Feature-level visualization ----
    plt.figure(figsize=(12, 6))
    plt.imshow(node_attributions, aspect='auto', cmap='RdBu_r')
    plt.colorbar(label="Attribution Value")
    plt.xlabel("Feature Index")
    plt.ylabel("Node Index")
    plt.title("Integrated Gradients: Feature Attributions per Node", fontsize=14)
    plt.show()

    # ---- Save results ----
    np.save(os.path.join(output_path, 'node_importance.npy'), node_importance)
    plt.savefig(os.path.join(output_path, "node_importance_heatmap.png"), dpi=300)
    print(f"Saved node importance and heatmap to {output_path}")



    # # ==============================
    # # 8. Top-10 Most Important Nodes (with labels)
    # # ==============================
    # # If your graph has pathway/protein names stored in ndata, fetch them:
    # if 'name' in G_dgl.ndata:
    #     node_names = [str(name) for name in G_dgl.ndata['name'].tolist()]
    # else:
    #     # fallback: just use node indices
    #     node_names = [f"Node {i}" for i in range(G_dgl.num_nodes())]

    # # Rank nodes by importance
    # topk = 10
    # top_indices = np.argsort(node_importance)[::-1][:topk]

    # print(f"\nTop-{topk} most important nodes:")
    # for rank, idx in enumerate(top_indices, start=1):
    #     print(f"{rank}. {node_names[idx]} (Index: {idx}, Score: {node_importance[idx]:.4f})")

def train_and_evaluate__(args, G_dgl, node_features):
    # ----------------------------
    # 1. Split edges into train/val/test
    # ----------------------------
    u, v = G_dgl.edges()
    eids = np.random.permutation(np.arange(G_dgl.number_of_edges()))
    test_size = int(len(eids) * 0.1)
    val_size = int(len(eids) * 0.1)

    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    val_pos_u, val_pos_v = u[eids[test_size:test_size + val_size]], v[eids[test_size:test_size + val_size]]
    train_pos_u, train_pos_v = u[eids[test_size + val_size:]], v[eids[test_size + val_size:]]

    # Negative edges
    adj = sp.coo_matrix(
        (np.ones(len(u)), (u.numpy(), v.numpy())),
        shape=(G_dgl.number_of_nodes(), G_dgl.number_of_nodes())
    )
    adj_neg = 1 - adj.todense() - np.eye(G_dgl.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    neg_eids = np.random.choice(len(neg_u), G_dgl.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    val_neg_u, val_neg_v = neg_u[neg_eids[test_size:test_size + val_size]], neg_v[neg_eids[test_size:test_size + val_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size + val_size:]], neg_v[neg_eids[test_size + val_size:]]

    # Training graph without test/val edges
    train_g = dgl.remove_edges(G_dgl, eids[:test_size + val_size])

    def create_graph(u, v, num_nodes):
        return dgl.graph((u, v), num_nodes=num_nodes)

    train_pos_g = create_graph(train_pos_u, train_pos_v, G_dgl.number_of_nodes())
    train_neg_g = create_graph(train_neg_u, train_neg_v, G_dgl.number_of_nodes())
    val_pos_g = create_graph(val_pos_u, val_pos_v, G_dgl.number_of_nodes())
    val_neg_g = create_graph(val_neg_u, val_neg_v, G_dgl.number_of_nodes())
    test_pos_g = create_graph(test_pos_u, test_pos_v, G_dgl.number_of_nodes())
    test_neg_g = create_graph(test_neg_u, test_neg_v, G_dgl.number_of_nodes())

    # ----------------------------
    # 2. Initialize model, predictor, loss, optimizer
    # ----------------------------
    hidden_feats = getattr(args, "hidden_feats", 64)
    out_feats = getattr(args, "out_feats", 2)

    model = GATModel(
        node_features.shape[1],
        hidden_feats=hidden_feats,
        out_feats=out_feats,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        feat_drop=0.2,
        attn_drop=0.2,
    )
    pred = MLPPredictor(args.input_size, args.hidden_size)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)

    output_path = './prediction/results/'
    os.makedirs(output_path, exist_ok=True)

    epoch_metrics = {
        'train': {'F1': [], 'AUC': [], 'Precision': [], 'Recall': [], 'FocalLoss': []},
        'val': {'F1': [], 'AUC': [], 'Precision': [], 'Recall': [], 'FocalLoss': []}
    }

    # ----------------------------
    # 3. Training loop
    # ----------------------------
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

        if e % 5 == 0:
            print(f'Epoch {e}, loss: {loss.item():.4f}')

        with torch.no_grad():
            # Train metrics
            h_train = model(train_g, train_g.ndata['feat'])
            train_pos_score = pred(train_pos_g, h_train)
            train_neg_score = pred(train_neg_g, h_train)
            epoch_metrics['train']['F1'].append(compute_f1(train_pos_score, train_neg_score).item())
            epoch_metrics['train']['AUC'].append(compute_auc(train_pos_score, train_neg_score).item())
            epoch_metrics['train']['Precision'].append(compute_precision(train_pos_score, train_neg_score))
            epoch_metrics['train']['Recall'].append(compute_recall(train_pos_score, train_neg_score))
            epoch_metrics['train']['FocalLoss'].append(compute_focalloss(train_pos_score, train_neg_score))

            # Validation metrics
            h_val = model(train_g, train_g.ndata['feat'])
            val_pos_score = pred(val_pos_g, h_val)
            val_neg_score = pred(val_neg_g, h_val)
            epoch_metrics['val']['F1'].append(compute_f1(val_pos_score, val_neg_score).item())
            epoch_metrics['val']['AUC'].append(compute_auc(val_pos_score, val_neg_score).item())
            epoch_metrics['val']['Precision'].append(compute_precision(val_pos_score, val_neg_score))
            epoch_metrics['val']['Recall'].append(compute_recall(val_pos_score, val_neg_score))
            epoch_metrics['val']['FocalLoss'].append(compute_focalloss(val_pos_score, val_neg_score))

    # ----------------------------
    # 4. Test evaluation
    # ----------------------------
    with torch.no_grad():
        model.eval()
        h_test = model(G_dgl, G_dgl.ndata['feat'])
        test_pos_score = pred(test_pos_g, h_test)
        test_neg_score = pred(test_neg_g, h_test)

        test_auc, test_auc_err = compute_auc_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_f1, test_f1_err = compute_f1_with_symmetrical_confidence(test_pos_score, test_neg_score)

        print(f'Test AUC: {test_auc:.4f} ± {test_auc_err:.4f} | Test F1: {test_f1:.4f} ± {test_f1_err:.4f}')

    # Save predictor model
    torch.save(pred.state_dict(), os.path.join(output_path, 'pred_model.pth'))

    # ----------------------------
    # 5. Node importance & spectral biclustering
    # ----------------------------
    node_attributions = h_test.detach().cpu().numpy()
    node_importance = np.linalg.norm(node_attributions, axis=1)

    # Save node importance
    np.save(os.path.join(output_path, 'node_importance.npy'), node_importance)

    # ---- Spectral biclustering ----
    n_row_clusters = 4
    n_col_clusters = 4
    bicluster = SpectralBiclustering(
        n_clusters=(n_row_clusters, n_col_clusters),
        method='log',
        random_state=42
    )
    bicluster.fit(node_attributions)
    row_labels = bicluster.row_labels_
    col_labels = bicluster.column_labels_

    # Save cluster labels
    np.save(os.path.join(output_path, 'node_cluster_labels.npy'), row_labels)
    np.save(os.path.join(output_path, 'feature_cluster_labels.npy'), col_labels)

    # ---- Heatmap visualization ----
    ordered_rows = np.argsort(row_labels)
    ordered_cols = np.argsort(col_labels)
    clustered_matrix = node_attributions[ordered_rows, :][:, ordered_cols]

    plt.figure(figsize=(12, 6))
    sns.heatmap(clustered_matrix, cmap='coolwarm')
    plt.xlabel("Features (clustered)")
    plt.ylabel("Nodes (clustered)")
    plt.title("Spectral Biclustering of Node Feature Attributions")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "bicluster_heatmap.png"), dpi=300)
    plt.close()

    print(f"✅ Node importance and biclustering results saved to {output_path}")

    return epoch_metrics, row_labels, col_labels

# def aggregate_node_importance(node_attributions: np.ndarray) -> np.ndarray:
#     """
#     Aggregate node importance across features using absolute sum with a progress bar.

#     Args:
#         node_attributions (np.ndarray): Node attribution values, shape (N, F).

#     Returns:
#         np.ndarray: Importance score per node, shape (N,).
#     """
#     importance = []
#     for row in tqdm(node_attributions, desc="Aggregating node importance"):
#         importance.append(np.abs(row).sum())
#     return np.array(importance)


# def plot_graph_attributions(G_dgl, node_importance, title="Integrated Gradients: Node Feature Importance"):
#     """
#     Visualize node importance on the graph using NetworkX with progress feedback.

#     Args:
#         G_dgl (dgl.DGLGraph): Input graph.
#         node_importance (np.ndarray): Importance scores for nodes.
#         title (str): Plot title.
#     """
#     print("🔄 Normalizing importance scores ...")
#     node_importance_norm = (node_importance - node_importance.min()) / (
#         node_importance.max() - node_importance.min() + 1e-9
#     )

#     print("🔄 Building NetworkX graph ...")
#     G_nx = G_dgl.to_networkx().to_undirected()
#     pos = nx.spring_layout(G_nx, seed=42, progress=True)  # newer NetworkX supports progress

#     print("🎨 Plotting graph ...")
#     plt.figure(figsize=(10, 8))
#     nodes = nx.draw_networkx_nodes(
#         G_nx, pos,
#         node_size=100,
#         node_color=node_importance_norm,
#         cmap=plt.cm.viridis
#     )
#     nx.draw_networkx_edges(G_nx, pos, alpha=0.3)
#     plt.colorbar(nodes, label="Node Attribution Importance")
#     plt.title(title, fontsize=14)
#     plt.axis("off")
#     plt.show()


# def plot_feature_heatmap(node_attributions: np.ndarray, title="Integrated Gradients: Feature Attributions per Node"):
#     """
#     Plot heatmap of feature-level attributions per node.
#     """
#     print("🎨 Plotting feature-level heatmap ...")
#     plt.figure(figsize=(12, 6))
#     plt.imshow(node_attributions, aspect='auto', cmap='RdBu_r')
#     plt.colorbar(label="Attribution Value")
#     plt.xlabel("Feature Index")
#     plt.ylabel("Node Index")
#     plt.title(title, fontsize=14)
#     plt.show()

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx
# from tqdm import tqdm

# def aggregate_node_importance(node_attributions: np.ndarray) -> np.ndarray:
#     """
#     Aggregate node importance across features using absolute sum with a progress bar.
#     """
#     importance = []
#     for row in tqdm(node_attributions, desc="Aggregating node importance"):
#         importance.append(np.abs(row).sum())
#     return np.array(importance)

# def plot_graph_attributions(G_dgl, node_importance, title="Integrated Gradients: Node Feature Importance"):
#     """
#     Visualize node importance on the graph using NetworkX with progress feedback.
#     """
#     print("🔄 Normalizing importance scores ...")
#     node_importance_norm = (node_importance - node_importance.min()) / (
#         node_importance.max() - node_importance.min() + 1e-9
#     )

#     print("🔄 Building NetworkX graph ...")
#     G_nx = G_dgl.to_networkx().to_undirected()

#     # Manually wrap spring_layout iteration in tqdm
#     print("⚡ Computing layout (spring)...")
#     pos = nx.spring_layout(G_nx, seed=42, iterations=50, weight=None, k=None, progress=None)
#     # Note: networkx doesn't expose internal iteration → progress bar faked with tqdm
#     for _ in tqdm(range(50), desc="Layout iterations"):
#         pass

#     print("🎨 Plotting graph ...")
#     plt.figure(figsize=(10, 8))
#     nodes = nx.draw_networkx_nodes(
#         G_nx, pos,
#         node_size=100,
#         node_color=node_importance_norm,
#         cmap=plt.cm.viridis
#     )
#     nx.draw_networkx_edges(G_nx, pos, alpha=0.3)
#     plt.colorbar(nodes, label="Node Attribution Importance")
#     plt.title(title, fontsize=14)
#     plt.axis("off")
#     plt.show()

# def plot_feature_heatmap(node_attributions: np.ndarray, title="Integrated Gradients: Feature Attributions per Node"):
#     """
#     Plot heatmap of feature-level attributions per node.
#     """
#     print("🎨 Plotting feature-level heatmap ...")
#     plt.figure(figsize=(12, 6))
#     plt.imshow(node_attributions, aspect='auto', cmap='RdBu_r')
#     plt.colorbar(label="Attribution Value")
#     plt.xlabel("Feature Index")
#     plt.ylabel("Node Index")
#     plt.title(title, fontsize=14)
#     plt.show()

# def save_node_importance(node_importance: np.ndarray, output_path: str):
#     """
#     Save node importance scores and corresponding heatmap with progress feedback.
#     """
#     print("💾 Saving node importance ...")
#     os.makedirs(output_path, exist_ok=True)

#     np.save(os.path.join(output_path, 'node_importance.npy'), node_importance)

#     print("📊 Saving barplot ...")
#     for _ in tqdm(range(1), desc="Saving plots"):
#         plt.figure(figsize=(12, 6))
#         plt.bar(np.arange(len(node_importance)), node_importance)
#         plt.xlabel("Node Index")
#         plt.ylabel("Importance Score")
#         plt.title("Node Importance Scores")
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_path, "node_importance_barplot.png"), dpi=300)
#         plt.close()

#     print(f"✅ Saved node importance and barplot to {output_path}")

# def print_top_nodes(node_importance: np.ndarray, G_dgl, topk: int = 10):
#     """
#     Print the top-k most important nodes with progress bar.
#     """
#     print(f"🏆 Ranking Top-{topk} nodes ...")

#     if 'name' in G_dgl.ndata:
#         node_names = [str(name) for name in G_dgl.ndata['name'].tolist()]
#     else:
#         node_names = [f"Node {i}" for i in range(G_dgl.num_nodes())]

#     top_indices = np.argsort(node_importance)[::-1][:topk]

#     for rank, idx in enumerate(tqdm(top_indices, desc="Printing top nodes"), start=1):
#         print(f"{rank}. {node_names[idx]} (Index: {idx}, Score: {node_importance[idx]:.4f})")




# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx
# import dgl

# ==============================
# 📊 Plot per-epoch metrics
# ==============================
def plot_epoch_metrics(epoch_metrics, output_path, args):
    """
    Plot training/validation metrics over epochs.
    Handles nested dict structure (train/val -> metrics).
    """
    os.makedirs(output_path, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for split, metrics in epoch_metrics.items():   # split = "train" or "val"
        for metric_name, values in metrics.items():
            plt.plot(values, label=f"{split}_{metric_name}")
            if len(values) > 0:
                print(f"{split}_{metric_name}: first element = {values[0]}")

    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.title("Training / Validation Metrics")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "epoch_metrics.png"))
    plt.close()

# ==============================
# 🔎 Attribution aggregation
# ==============================
def aggregate_node_importance(node_attributions):
    """
    Aggregate node attributions into an importance score.
    Example: L2 norm across features.
    """
    importance = np.linalg.norm(node_attributions, axis=1)
    return importance

# def save_node_importance(node_importance, output_path):
#     """
#     Save node importance scores to a CSV file.
#     """
#     os.makedirs(output_path, exist_ok=True)
#     out_file = os.path.join(output_path, "node_importance.csv")

#     np.savetxt(out_file, node_importance, delimiter=",")
#     print(f"✅ Node importance saved to {out_file}")

# ==============================
# 🎨 Attribution Visualizations
# ==============================

def plot_graph_attributions(G_dgl, node_importance, output_path, topk=1000):
    """
    Plot graph with node importance as color/size (restricted to top-k nodes).
    """
    os.makedirs(output_path, exist_ok=True)

    G_nx = G_dgl.to_networkx()

    # Select top-k nodes by importance
    topk = min(topk, len(node_importance))
    top_indices = np.argsort(node_importance)[-topk:]

    sub_nodes = [list(G_nx.nodes())[i] for i in top_indices]
    G_sub = G_nx.subgraph(sub_nodes)

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    pos = nx.spring_layout(G_sub, seed=42)

    # Normalize importance for color mapping
    node_imp_sub = node_importance[top_indices]
    norm = plt.Normalize(vmin=node_imp_sub.min(), vmax=node_imp_sub.max())
    cmap = plt.cm.viridis

    node_size = 300 + 2000 * (node_imp_sub / node_imp_sub.max())
    nx.draw(
        G_sub,
        pos,
        with_labels=True,
        node_color=node_imp_sub,
        cmap=cmap,
        node_size=node_size,
        font_size=8,
        ax=ax
    )

    plt.title(f"Node Importance Visualization (Top {topk})")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Importance")

    plt.tight_layout()
    out_file = os.path.join(output_path, f"graph_attributions_top{topk}.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"✅ Saved graph attribution plot to {out_file}")

def plot_feature_heatmap(node_attributions, output_path="plots", topk=1000):
    """
    Plot feature heatmap for top-k nodes by importance.
    """
    os.makedirs(output_path, exist_ok=True)

    # Compute node importance as L1 norm across features
    node_importance = np.abs(node_attributions).sum(axis=1)

    # Select top-k nodes
    topk = min(topk, len(node_importance))
    top_indices = np.argsort(node_importance)[-topk:]
    top_node_attributions = node_attributions[top_indices]

    plt.figure(figsize=(12, 6))
    plt.imshow(top_node_attributions, aspect="auto", cmap="coolwarm")
    plt.colorbar(label="Attribution Value")
    plt.xlabel("Features")
    plt.ylabel("Top Nodes")
    plt.title(f"Node Feature Attribution Heatmap (Top {topk})")
    plt.tight_layout()
    out_file = os.path.join(output_path, f"feature_heatmap_top{topk}.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"✅ Saved feature heatmap to {out_file}")

def print_top_nodes(node_importance, G_dgl, topk=10):
    """
    Print top-k nodes by importance.
    """
    G_nx = G_dgl.to_networkx()
    node_list = list(G_nx.nodes)

    top_indices = np.argsort(-node_importance)[:topk]
    print(f"\n🏆 Top-{topk} important nodes:")
    for idx in top_indices:
        print(f"Node {node_list[idx]} → importance {node_importance[idx]:.4f}")

def save_node_importance(node_importance: np.ndarray, output_path: str):
    """
    Save node importance scores to CSV and NPY,
    and generate a barplot with progress feedback.
    """
    print("💾 Saving node importance ...")
    os.makedirs(output_path, exist_ok=True)

    # Save CSV
    csv_file = os.path.join(output_path, "node_importance.csv")
    np.savetxt(csv_file, node_importance, delimiter=",")
    
    # Save NPY
    npy_file = os.path.join(output_path, "node_importance.npy")
    np.save(npy_file, node_importance)

    print("📊 Generating plots ...")
    for _ in tqdm(range(1), desc="Saving plots"):
        plt.figure(figsize=(12, 6))
        plt.bar(np.arange(len(node_importance)), node_importance, color="skyblue")
        plt.xlabel("Node Index")
        plt.ylabel("Importance Score")
        plt.title("Node Importance Scores")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "node_importance_barplot.png"), dpi=300)
        plt.close()

    print(f"✅ Node importance saved to:\n   - {csv_file}\n   - {npy_file}\n   - {output_path}/node_importance_barplot.png")

def plot_feature_heatmap(node_attributions: np.ndarray, output_path: str, topk: int = 1000):
    """
    Plot feature heatmap for top-k predicted nodes (cell-based format).
    Saves to output_path.
    """
    print("📊 Generating feature attribution heatmap ...")
    os.makedirs(output_path, exist_ok=True)

    # Restrict to top-k nodes
    if node_attributions.shape[0] > topk:
        node_attributions = node_attributions[:topk, :]

    for _ in tqdm(range(1), desc="Saving heatmap"):
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(
            node_attributions,
            cmap="coolwarm",
            cbar_kws={"label": "Attribution Value"},
            square=False,
            linewidths=0.3,   # 👈 adds cell borders
            linecolor="gray"
        )
        ax.set_xlabel("Features")
        ax.set_ylabel("Nodes")
        ax.set_title(f"Node Feature Attribution Heatmap (Top {topk})")

        plt.tight_layout()
        out_file = os.path.join(output_path, f"feature_heatmap_top{topk}.png")
        plt.savefig(out_file, dpi=300)
        plt.close()

    print(f"✅ Feature heatmap saved to: {out_file}")

# ==============================
# 🚀 MAIN CALL
# ==============================
def run_post_training_analysis(epoch_metrics, h_test, pred, G_dgl, output_path, args):
    # ---- Plot per-epoch metrics ----
    plot_epoch_metrics(epoch_metrics, output_path, args)

    # ==============================
    # 🔎 Attribution + Visualization
    # ==============================
    print("⚡ Computing node importance with attributions ...")
    node_attributions = h_test.detach().cpu().numpy()   # shape (N, F)

    node_importance = aggregate_node_importance(node_attributions)

    # Save results and plots
    save_node_importance(node_importance, output_path)
    plot_graph_attributions(G_dgl, node_importance, output_path)
    plot_feature_heatmap(node_attributions, output_path)

    # Print top-10 nodes
    print_top_nodes(node_importance, G_dgl, topk=100)

    # ==============================
    # 💾 Save predictor model
    # ==============================
    torch.save(pred.state_dict(), os.path.join(output_path, 'pred_model.pth'))
    print(f"✅ Model saved at {os.path.join(output_path, 'pred_model.pth')}")

def plot_node_importance_graph(G_dgl, node_attributions, output_path):
    """Plot and save node importance visualization on the graph."""
    os.makedirs(output_path, exist_ok=True)

    # Aggregate importance per node
    node_importance = np.abs(node_attributions).sum(axis=1)

    # Normalize for coloring
    node_importance_norm = (node_importance - node_importance.min()) / (node_importance.max() - node_importance.min() + 1e-9)

    # Build NetworkX graph
    G_nx = G_dgl.to_networkx().to_undirected()
    pos = nx.spring_layout(G_nx, seed=42)

    # Plot node importance on graph
    plt.figure(figsize=(10, 8))
    nodes = nx.draw_networkx_nodes(
        G_nx, pos,
        node_size=100,
        node_color=node_importance_norm,
        cmap=plt.cm.viridis
    )
    nx.draw_networkx_edges(G_nx, pos, alpha=0.3)
    plt.colorbar(nodes, label="Node Attribution Importance")
    plt.title("Integrated Gradients: Node Feature Importance", fontsize=14)
    plt.axis("off")
    plt.savefig(os.path.join(output_path, "node_importance_graph.png"), dpi=300)
    plt.close()

    # Save numeric importance
    np.save(os.path.join(output_path, 'node_importance.npy'), node_importance)
    print(f"Saved node importance graph + values to {output_path}")

def plot_node_feature_heatmap(node_attributions, output_path):
    """Plot and save feature-level attribution heatmap."""
    os.makedirs(output_path, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.imshow(node_attributions, aspect='auto', cmap='RdBu_r')
    plt.colorbar(label="Attribution Value")
    plt.xlabel("Feature Index")
    plt.ylabel("Node Index")
    plt.title("Integrated Gradients: Feature Attributions per Node", fontsize=14)
    plt.savefig(os.path.join(output_path, "node_feature_heatmap.png"), dpi=300)
    plt.close()

    print(f"Saved node feature attribution heatmap to {output_path}")

def plot_spectral_biclustering(node_attributions, output_path, topk=1000, n_clusters_row=4, n_clusters_col=4):
    """Apply spectral biclustering on top-k nodes and save heatmap with cluster boundaries."""
    os.makedirs(output_path, exist_ok=True)

    # Compute importance and select top-k nodes
    node_importance = np.abs(node_attributions).sum(axis=1)
    top_indices = np.argsort(node_importance)[::-1][:min(topk, len(node_importance))]
    top_node_attr = node_attributions[top_indices, :]

    # Apply spectral biclustering
    bicluster = SpectralBiclustering(n_clusters=(n_clusters_row, n_clusters_col), random_state=42)
    bicluster.fit(np.abs(top_node_attr))

    # Reorder rows and columns
    fit_data = top_node_attr[np.argsort(bicluster.row_labels_)]
    fit_data = fit_data[:, np.argsort(bicluster.column_labels_)]

    # Plot heatmap with cluster boundaries
    plt.figure(figsize=(12, 8))
    plt.imshow(fit_data, aspect='auto', cmap='RdBu_r')
    plt.colorbar(label="Attribution Value")
    plt.xlabel("Feature Clustered Index")
    plt.ylabel(f"Top {topk} Node Clustered Index")
    plt.title("Spectral Biclustering: Node-Feature Attribution Heatmap")

    row_splits = np.cumsum(np.bincount(bicluster.row_labels_)[np.argsort(np.unique(bicluster.row_labels_))])
    col_splits = np.cumsum(np.bincount(bicluster.column_labels_)[np.argsort(np.unique(bicluster.column_labels_))])

    for r in row_splits[:-1]:
        plt.axhline(r - 0.5, color='black', linewidth=1.2)
    for c in col_splits[:-1]:
        plt.axvline(c - 0.5, color='black', linewidth=1.2)

    plt.savefig(os.path.join(output_path, "spectral_bicluster_heatmap.png"), dpi=300)
    plt.close()

    print(f"Saved spectral biclustering heatmap (top {topk} nodes) to {output_path}")

def train_and_evaluate_ori(args, G_dgl, node_features):
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

    def create_graph(u, v, num_nodes):
        assert len(u) == len(v), "Source and destination nodes must have the same length"
        return dgl.graph((u, v), num_nodes=num_nodes)

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
    
    
def save_top_pathway_predictions(preds, edge_index, pathway_names, output_dir="results", top_k=1000, labels=None):
    """
    Save top-K predicted pathway associations to a CSV file.

    Args:
        preds (torch.Tensor or np.ndarray): Prediction scores for pathway pairs.
        edge_index (torch.Tensor or np.ndarray): Edge indices (2 x num_edges).
        pathway_names (list or np.ndarray): Names of pathways in the graph.
        output_dir (str): Directory where CSV will be saved.
        top_k (int): Number of top predictions to save.
        labels (torch.Tensor or np.ndarray, optional): Ground-truth labels for edges.

    Returns:
        pd.DataFrame: DataFrame of top-K predictions.
    """
    # Convert tensors to numpy
    if not isinstance(preds, np.ndarray):
        preds = preds.detach().cpu().numpy()
    if not isinstance(edge_index, np.ndarray):
        edge_index = edge_index.detach().cpu().numpy()
    if labels is not None and not isinstance(labels, np.ndarray):
        labels = labels.detach().cpu().numpy()

    # Sort by prediction score (descending)
    sorted_idx = np.argsort(-preds)[:top_k]

    # Build DataFrame
    src_pathways = [pathway_names[edge_index[0, i]] for i in sorted_idx]
    dst_pathways = [pathway_names[edge_index[1, i]] for i in sorted_idx]
    scores = preds[sorted_idx]

    data = {
        "Rank": range(1, len(sorted_idx) + 1),
        "Source_Pathway": src_pathways,
        "Target_Pathway": dst_pathways,
        "Prediction_Score": scores,
    }

    if labels is not None:
        data["True_Label"] = labels[sorted_idx]

    df = pd.DataFrame(data)

    # Save CSV
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"top_{top_k}_pathway_predictions.csv")
    df.to_csv(out_path, index=False)

    print(f"✅ Saved top {top_k} pathway predictions to {out_path}")
    return df

def train_and_evaluate_10x10(args, G_dgl, node_features):
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

    def create_graph(u, v, num_nodes):
        assert len(u) == len(v), "Source and destination nodes must have the same length"
        return dgl.graph((u, v), num_nodes=num_nodes)

    train_pos_g = create_graph(train_pos_u, train_pos_v, G_dgl.number_of_nodes())
    train_neg_g = create_graph(train_neg_u, train_neg_v, G_dgl.number_of_nodes())
    val_pos_g = create_graph(val_pos_u, val_pos_v, G_dgl.number_of_nodes())
    val_neg_g = create_graph(val_neg_u, val_neg_v, G_dgl.number_of_nodes())
    test_pos_g = create_graph(test_pos_u, test_pos_v, G_dgl.number_of_nodes())
    test_neg_g = create_graph(test_neg_u, test_neg_v, G_dgl.number_of_nodes())
    hidden_feats = getattr(args, "hidden_feats", 64)
    out_feats = getattr(args, "out_feats", 2)

    # model = GATModel(
    #     node_features.shape[1],
    #     hidden_feats=hidden_feats,
    #     out_feats=out_feats,
    #     num_layers=args.num_layers,
    #     num_heads=args.num_heads,
    #     feat_drop=0.2,
    #     attn_drop=0.2,
    # )

    in_feats = node_features.shape[1]
    model = ECGNN(in_feats, hidden_feats=64, out_feats=out_feats, k=3)

    # model = ECGNN(
    #     in_feats=node_features.shape[1],
    #     hidden_feats=hidden_feats,
    #     out_feats=out_feats,
    #     # k=k_order,
    #     # dropout=dropout,
    #     # epsilon=epsilon
    # )

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
    src, dst = int(test_pos_u[0]), int(test_pos_v[0])  # pick a test pair
    explanation = explain_pathway_link_dgl(
        model=model,
        G_dgl=G_dgl,
        x=G_dgl.ndata['feat'],   # pass node features
        src=src,
        dst=dst,
        node_names={i: f"Pathway_{i}" for i in range(G_dgl.num_nodes())},
        top_k=15
    )

    # the actual NetworkX subgraph:
    H = explanation["graph"]
    output_file = "prediction/results/reactome_link_explanation_node1935.png"
    plot_explanation(H, explanation, output_file=output_file)



def plot_explanation(H, explanation, output_file=None):
    """
    Plot the explanation subgraph with node importance and save to output_file if given.
    """
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(H, seed=42)  # deterministic layout

    # Draw nodes with importance
    if "importance" in explanation:
        node_imp = [explanation["importance"].get(n, 0.1) for n in H.nodes()]
        nx.draw_networkx_nodes(
            H,
            pos,
            node_size=[50 * v for v in node_imp],  # size by importance
            node_color=node_imp,                    # color by importance
            cmap=plt.cm.Reds,
            alpha=0.6
        )
    else:
        nx.draw_networkx_nodes(H, pos, node_size=100, node_color="lightblue")

    # Draw edges
    nx.draw_networkx_edges(H, pos, alpha=0.5)

    # Labels (prefer human-readable names if available)
    labels = {n: explanation.get("names", {}).get(n, str(n)) for n in H.nodes()}
    nx.draw_networkx_labels(H, pos, labels, font_size=8)

    # Colorbar for node importance
    if "importance" in explanation:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                                   norm=plt.Normalize(vmin=min(node_imp), vmax=max(node_imp)))
        sm.set_array([])
        plt.colorbar(sm, label="Node importance")

    plt.axis("off")

    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"[Info] Saved explanation plot to {output_file}")

    plt.show()

def explain_pathway_link_dgl(model, G_dgl, x, src, dst, node_names=None, top_k=15):
    """
    Explain why a link exists between src and dst.
    Returns explanation masks (src, dst, combined) and the subgraph for visualization.
    """
    try:
        model.eval()
        explainer = GNNExplainer(model, num_hops=3)

        # ---- Source ----
        src_result = explainer.explain_node(src, G_dgl, x)
        if isinstance(src_result, tuple) and len(src_result) >= 2:
            src_feat_mask, src_edge_mask = src_result[:2]
        else:
            src_feat_mask, src_edge_mask = None, src_result  # fallback

        # ---- Destination ----
        dst_result = explainer.explain_node(dst, G_dgl, x)
        if isinstance(dst_result, tuple) and len(dst_result) >= 2:
            dst_feat_mask, dst_edge_mask = dst_result[:2]
        else:
            dst_feat_mask, dst_edge_mask = None, dst_result  # fallback

        # ---- Ensure edge masks are tensors ----
        if isinstance(src_edge_mask, dgl.DGLGraph):
            src_edge_mask = torch.ones(G_dgl.number_of_edges())
        if isinstance(dst_edge_mask, dgl.DGLGraph):
            dst_edge_mask = torch.ones(G_dgl.number_of_edges())

        # ---- Combine ----
        combined_edge_mask = (src_edge_mask + dst_edge_mask) / 2

        # ---- Build subgraph for visualization ----
        edges = G_dgl.edges(order="eid")
        edge_index = torch.stack(edges).t().tolist()

        G = nx.Graph()
        for i, (u, v) in enumerate(edge_index):
            weight = combined_edge_mask[i].item()
            if weight > 0:
                G.add_edge(u, v, weight=weight)

        labels = {i: node_names[i] if node_names else str(i) for i in G.nodes()}
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=600)
        nx.draw_networkx_edges(
            G, pos, width=2, alpha=0.6, edge_color="red",
            edgelist=G.edges(), 
            connectionstyle="arc3,rad=0.1"
        )
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        plt.title(f"Explanation Subgraph: {labels.get(src, src)} ↔ {labels.get(dst, dst)}")
        plt.axis("off")
        plt.show()

        return {
            "src_edge_mask": src_edge_mask,
            "dst_edge_mask": dst_edge_mask,
            "combined_edge_mask": combined_edge_mask,
            "graph": G
        }

    except Exception as e:
        print(f"[Warning] Failed to explain link {src} ↔ {dst}: {e}")
        return {
            "src_edge_mask": None,
            "dst_edge_mask": None,
            "combined_edge_mask": None,
            "graph": None
        }

