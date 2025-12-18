import json
import os
from matplotlib import pyplot as plt
import torch
import itertools
import dgl
import numpy as np
import scipy.sparse as sp
from .models import GCNModel, ChebNetModel, MLPPredictor, FocalLoss
from .utils import compute_loss, compute_hits_k, compute_auc, compute_f1, compute_focalloss, compute_accuracy, compute_precision, compute_recall, compute_map, compute_focalloss_with_symmetrical_confidence, compute_auc_with_symmetrical_confidence, compute_f1_with_symmetrical_confidence, compute_accuracy_with_symmetrical_confidence, compute_precision_with_symmetrical_confidence, compute_recall_with_symmetrical_confidence, compute_map_with_symmetrical_confidence
from scipy.stats import sem

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

    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
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
    
    # model = GCNModel(
    #     node_features.shape[1], 
    #     dim_latent=args.dim_latent, 
    #     num_layers=args.num_layers, 
    #     do_train=True
    # )

    in_feats = node_features.shape[1]
    dim_latent = args.dim_latent

    model = ChebNetModel(
        graph=G_dgl,
        in_feats=in_feats,
        dim_latent=dim_latent,
        num_layers=args.num_layers,
        k=3,
        do_train=True
    )


    pred = MLPPredictor(args.input_size, args.hidden_size)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')

    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)

    output_path = './link_prediction_gcn/results/'
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
  
    # def plot_scores(epochs, train_f1_scores, val_f1_scores, train_focal_loss_scores, val_focal_loss_scores, train_auc_scores, val_auc_scores, 
    #                 train_map_scores, val_map_scores, train_recall_scores, val_recall_scores,
    #                 train_acc_scores, val_acc_scores, train_precision_scores, val_precision_scores,
    #                 output_path, args):

    #     # Ensure the output directory exists
    #     os.makedirs(output_path, exist_ok=True)

    #     ##plt.figure(figsize=(15, 5))

    #     ##plt.subplot(1, 2, 1)
    #     plt.figure()
    #     plt.plot(epochs, train_f1_scores, label='Training F1 Score')
    #     plt.plot(epochs, val_f1_scores, label='Validation F1 Score')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('F1 Score')
    #     plt.title('Training and Validation F1 Scores over Epochs')
    #     plt.legend()
    #     ##plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #     plt.savefig(os.path.join(output_path, f'f1_dim{args.dim_latent}_lay{args.num_layers}_epo{args.epochs}.png'))

    #     ##plt.figure(figsize=(15, 5))

    #     ##plt.subplot(1, 2, 1)
    #     plt.figure()
    #     plt.plot(epochs, train_focal_loss_scores, label='Training FocalLoss Score')
    #     plt.plot(epochs, val_focal_loss_scores, label='Validation FocalLoss Score')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('FocalLoss Score')
    #     plt.title('Training and Validation FocalLoss Scores over Epochs')
    #     plt.legend()
    #     ##plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #     plt.savefig(os.path.join(output_path, f'loss_dim{args.dim_latent}_lay{args.num_layers}_epo{args.epochs}.png'))
        
    #     ##plt.figure(figsize=(15, 5))

    #     ##plt.subplot(1, 2, 1)
    #     plt.figure()
    #     plt.plot(epochs, train_auc_scores, label='Training AUC')
    #     plt.plot(epochs, val_auc_scores, label='Validation AUC')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('AUC')
    #     plt.title('Training and Validation AUC over Epochs')
    #     plt.legend()
    #     ##plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #     plt.savefig(os.path.join(output_path, f'auc_dim{args.dim_latent}_lay{args.num_layers}_epo{args.epochs}.png'))

    #     ##plt.figure(figsize=(15, 5))

    #     ##plt.subplot(1, 2, 1)
    #     plt.figure()
    #     plt.plot(epochs, train_map_scores, label='Training mAP')
    #     plt.plot(epochs, val_map_scores, label='Validation mAP')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('mAP')
    #     plt.title('Training and Validation mAP over Epochs')
    #     plt.legend()
    #     ##plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #     plt.savefig(os.path.join(output_path, f'mAP_dim{args.dim_latent}_lay{args.num_layers}_epo{args.epochs}.png'))


    #     ##plt.figure(figsize=(15, 5))

    #     ##plt.subplot(1, 2, 1)
    #     plt.figure()
    #     plt.plot(epochs, train_recall_scores, label='Training Recall')
    #     plt.plot(epochs, val_recall_scores, label='Validation Recall')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Recall')
    #     plt.title('Training and Validation Recall over Epochs')
    #     plt.legend()
    #     ##plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #     plt.savefig(os.path.join(output_path, f'recall_dim{args.dim_latent}_lay{args.num_layers}_epo{args.epochs}.png'))


    #     ##plt.figure(figsize=(15, 5))

    #     ##plt.subplot(1, 2, 1)
    #     plt.figure()
    #     plt.plot(epochs, train_acc_scores, label='Training Accuracy')
    #     plt.plot(epochs, val_acc_scores, label='Validation Accuracy')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Accuracy')
    #     plt.title('Training and Validation Accuracy over Epochs')
    #     plt.legend()
    #     ##plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #     plt.savefig(os.path.join(output_path, f'acc_dim{args.dim_latent}_lay{args.num_layers}_epo{args.epochs}.png'))
    #     ##plt.figure(figsize=(15, 5))

    #     ##plt.subplot(1, 2, 1)
    #     plt.figure()
    #     plt.plot(epochs, train_precision_scores, label='Training Precision')
    #     plt.plot(epochs, val_precision_scores, label='Validation Precision')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Precision')
    #     plt.title('Training and Validation Precision over Epochs')
    #     plt.legend()
    #     ##plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #     plt.savefig(os.path.join(output_path, f'precision_dim{args.dim_latent}_lay{args.num_layers}_epo{args.epochs}.png'))

    #     plt.show()

    def plot_metric(epochs, train_vals, val_vals, ylabel, title, save_path, log_scale=False):
        """
        Reusable helper function to generate clean metric plots.
        """
        plt.figure(figsize=(10, 8))

        if log_scale:
            plt.yscale("log")

        plt.plot(epochs, train_vals, linewidth=2.2, label=f'Train {ylabel}')
        plt.plot(epochs, val_vals, linewidth=2.2, label=f'Val {ylabel}')

        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16, pad=15)
        plt.legend(frameon=False, fontsize=12)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()



    def plot_scores(
            epochs,
            train_f1_scores, val_f1_scores,
            train_focal_loss_scores, val_focal_loss_scores,
            train_auc_scores, val_auc_scores,
            train_map_scores, val_map_scores,
            train_recall_scores, val_recall_scores,
            train_acc_scores, val_acc_scores,
            train_precision_scores, val_precision_scores,
            output_path,
            args
    ):
        os.makedirs(output_path, exist_ok=True)

        prefix = f"dim{args.dim_latent}_lay{args.num_layers}_epo{args.epochs}"

        # --- F1 Score ---
        plot_metric(
            epochs,
            train_f1_scores, val_f1_scores,
            ylabel="F1 Score",
            title="Training and Validation F1 Score",
            save_path=os.path.join(output_path, f"f1_{prefix}.png")
        )

        # --- Focal Loss (log scale recommended) ---
        plot_metric(
            epochs,
            train_focal_loss_scores, val_focal_loss_scores,
            ylabel="Focal Loss",
            title="Training and Validation Focal Loss",
            save_path=os.path.join(output_path, f"loss_{prefix}.png"),
            log_scale=True
        )

        # --- AUC ---
        plot_metric(
            epochs,
            train_auc_scores, val_auc_scores,
            ylabel="AUC",
            title="Training and Validation AUC",
            save_path=os.path.join(output_path, f"auc_{prefix}.png")
        )

        # --- mAP ---
        plot_metric(
            epochs,
            train_map_scores, val_map_scores,
            ylabel="mAP",
            title="Training and Validation mAP",
            save_path=os.path.join(output_path, f"map_{prefix}.png")
        )

        # --- Recall ---
        plot_metric(
            epochs,
            train_recall_scores, val_recall_scores,
            ylabel="Recall",
            title="Training and Validation Recall",
            save_path=os.path.join(output_path, f"recall_{prefix}.png")
        )

        # --- Accuracy ---
        plot_metric(
            epochs,
            train_acc_scores, val_acc_scores,
            ylabel="Accuracy",
            title="Training and Validation Accuracy",
            save_path=os.path.join(output_path, f"acc_{prefix}.png")
        )

        # --- Precision ---
        plot_metric(
            epochs,
            train_precision_scores, val_precision_scores,
            ylabel="Precision",
            title="Training and Validation Precision",
            save_path=os.path.join(output_path, f"precision_{prefix}.png")
        )

        print(f"✓ All plots saved to: {output_path}")

    plot_scores(epochs, train_f1_scores, val_f1_scores, train_focal_loss_scores, val_focal_loss_scores, train_auc_scores, val_auc_scores, 
        train_map_scores, val_map_scores, train_recall_scores, val_recall_scores,
        train_acc_scores, val_acc_scores, train_precision_scores, val_precision_scores,
        output_path, args)

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

    model_path = './link_prediction_gcn/results/pred_model.pth'
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

    with open(os.path.join(output_path, 'test_results.json'), 'w') as f:
        json.dump(output, f)

    filename = f'test_results_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.dim_latent}_epoch{args.epochs}.json'
    
    test_results = {
        'Learning Rate': args.lr,
        'Epochs': args.epochs,
        'Input Features': args.input_size,
        'Output Features': args.dim_latent,
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

def train_and_evaluate(args, G_dgl, node_features):

    import os, json, itertools
    import numpy as np
    import torch
    import scipy.sparse as sp
    import dgl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import SpectralBiclustering

    ########################################
    # 1. DATA SPLIT (UNCHANGED)
    ########################################

    u, v = G_dgl.edges()
    eids = np.random.permutation(np.arange(G_dgl.number_of_edges()))

    test_size = int(len(eids) * 0.1)
    val_size = int(len(eids) * 0.1)

    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    val_pos_u, val_pos_v = u[eids[test_size:test_size + val_size]], v[eids[test_size:test_size + val_size]]
    train_pos_u, train_pos_v = u[eids[test_size + val_size:]], v[eids[test_size + val_size:]]

    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(G_dgl.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), G_dgl.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    val_neg_u, val_neg_v = neg_u[neg_eids[test_size:test_size + val_size]], neg_v[neg_eids[test_size:test_size + val_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size + val_size:]], neg_v[neg_eids[test_size + val_size:]]

    train_g = dgl.remove_edges(G_dgl, eids[:test_size + val_size])

    def create_graph(u, v):
        return dgl.graph((u, v), num_nodes=G_dgl.number_of_nodes())

    train_pos_g = create_graph(train_pos_u, train_pos_v)
    train_neg_g = create_graph(train_neg_u, train_neg_v)
    val_pos_g = create_graph(val_pos_u, val_pos_v)
    val_neg_g = create_graph(val_neg_u, val_neg_v)
    test_pos_g = create_graph(test_pos_u, test_pos_v)
    test_neg_g = create_graph(test_neg_u, test_neg_v)

    ########################################
    # 2. MODEL SETUP (UNCHANGED)
    ########################################

    model = ChebNetModel(
        graph=G_dgl,
        in_feats=node_features.shape[1],
        dim_latent=args.dim_latent,
        num_layers=args.num_layers,
        k=3,
        do_train=True
    )

    pred = MLPPredictor(args.input_size, args.hidden_size)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.Adam(
        itertools.chain(model.parameters(), pred.parameters()),
        lr=args.lr
    )

    output_path = "./link_prediction_gcn/results/"
    os.makedirs(output_path, exist_ok=True)

    ########################################
    # 3. TRAINING LOOP (UNCHANGED)
    ########################################

    for e in range(args.epochs):
        model.train()
        h = model(train_g, train_g.ndata['feat'])

        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)

        labels = torch.cat([
            torch.ones_like(pos_score),
            torch.zeros_like(neg_score)
        ])
        scores = torch.cat([pos_score, neg_score])

        loss = criterion(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print(f"Epoch {e}, Loss: {loss.item():.4f}")

    ########################################
    # 4. TESTING (UNCHANGED)
    ########################################

    model.eval()
    with torch.no_grad():
        h_test = model(G_dgl, G_dgl.ndata['feat'])

    ########################################
    # 5. SAVE NODE EMBEDDINGS
    ########################################

    torch.save(
        h_test.cpu(),
        os.path.join(output_path, "node_embeddings.pt")
    )

    ########################################
    # 6. NODE SALIENCY (GENE + PATHWAY)
    ########################################

    G_dgl.ndata['feat'].requires_grad_(True)
    h = model(G_dgl, G_dgl.ndata['feat'])
    score = h.norm(p=2, dim=1).sum()
    score.backward()

    node_saliency = G_dgl.ndata['feat'].grad.abs().sum(dim=1)

    torch.save(
        node_saliency.cpu(),
        os.path.join(output_path, "node_saliency.pt")
    )
    
    

    ########################################
    # 7. PATHWAY SALIENCY AGGREGATION
    ########################################

    # pathway_saliency = torch.zeros(len(pathway_names))

    # for g, pathways in gene_to_pathways.items():
    #     for p in pathways:
    #         pathway_saliency[p] += node_saliency[g]

    # torch.save(
    #     pathway_saliency.cpu(),
    #     os.path.join(output_path, "pathway_saliency.pt")
    # )

    ########################################
    # 8. EDGE-LEVEL ATTRIBUTION
    ########################################

    u_all, v_all = G_dgl.edges()
    edge_embed = torch.cat([h_test[u_all], h_test[v_all]], dim=1)

    W = list(pred.parameters())[0]
    edge_attr = torch.abs(edge_embed @ W.T).sum(dim=1)

    np.save(
        os.path.join(output_path, "edge_attributions.npy"),
        # edge_attr.cpu().numpy()
        edge_attr.detach().cpu().numpy()

    )

    torch.save(
        pred.state_dict(),
        os.path.join(output_path, "predictor_weights.pt")
    )

    # ########################################
    # # 9. SALIENCY × PATHWAY MATRIX
    # ########################################

    # saliency_matrix = np.zeros((len(pathway_names), len(gene_names)))

    # for p, genes in pathway_to_genes.items():
    #     for g in genes:
    #         saliency_matrix[p, g] = node_saliency[g].item()

    # np.save(
    #     os.path.join(output_path, "saliency_matrix.npy"),
    #     saliency_matrix
    # )

    ########################################
    # 9A. LOAD PATHWAY ENRICHMENT CSV
    ########################################

    import pandas as pd

    df_enrich = pd.read_csv(
        "data/gene_gene_pairs_with_pathwayA_enrichment_unique_20_limit.csv"
    )


    # Keep only significant associations
    df_enrich = df_enrich[df_enrich["significance"] == "significant"]

    # Convert p-value → enrichment score
    df_enrich["enrich_score"] = -np.log10(df_enrich["pvalue"])

    # Gene / pathway names
    gene_names = sorted(df_enrich["Gene2"].unique())
    pathway_names = sorted(df_enrich["PathwayB"].unique())

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    pathway_to_idx = {p: i for i, p in enumerate(pathway_names)}

    # Initialize enrichment matrix (genes × pathways)
    enrich_matrix = np.zeros((len(gene_names), len(pathway_names)))

    for _, row in df_enrich.iterrows():
        i = gene_to_idx[row["Gene2"]]
        j = pathway_to_idx[row["PathwayB"]]
        enrich_matrix[i, j] = row["enrich_score"]

    np.save(
        os.path.join(output_path, "gene_pathway_enrichment.npy"),
        enrich_matrix
    )

    ########################################
    # 9B. SALIENCY × ENRICHMENT MATRIX
    ########################################

    saliency_np = node_saliency.cpu().numpy()

    # Pathway × Gene matrix
    saliency_pathway_matrix = np.zeros(
        (len(pathway_names), len(gene_names))
    )

    for g_name, g_idx in gene_to_idx.items():
        for p_name, p_idx in pathway_to_idx.items():
            saliency_pathway_matrix[p_idx, g_idx] = (
                saliency_np[g_idx] * enrich_matrix[g_idx, p_idx]
            )

    np.save(
        os.path.join(output_path, "saliency_enrichment_matrix.npy"),
        saliency_pathway_matrix
    )

    ########################################
    # 10. HEATMAP
    ########################################

    plt.figure(figsize=(10, 8))
    sns.heatmap(saliency_pathway_matrix, cmap="Reds", yticklabels=pathway_names)
    plt.xlabel("Genes")
    plt.ylabel("Pathways")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "saliency_pathway_heatmap.png"))
    plt.close()

    ########################################
    # 11. SPECTRAL BICLUSTERING
    ########################################

    bicluster = SpectralBiclustering(
        n_clusters=(10, 10),
        method="log",
        random_state=0
    )
    bicluster.fit(saliency_pathway_matrix)

    np.save(
        os.path.join(output_path, "bicluster_labels.npy"),
        bicluster.row_labels_
    )

    ########################################
    # 12. GENE–PATHWAY EXPLANATION TABLE
    ########################################

    rows = []
    for p in range(saliency_pathway_matrix.shape[0]):
        top_genes = np.argsort(saliency_pathway_matrix[p])[::-1][:20]
        for g in top_genes:
            rows.append([
                pathway_names[p],
                gene_names[g],
                saliency_pathway_matrix[p, g]
            ])

    import pandas as pd
    df = pd.DataFrame(rows, columns=["Pathway", "Gene", "Saliency"])
    df.to_csv(
        os.path.join(output_path, "top_gene_pathway_pairs.csv"),
        index=False
    )

    print("✅ Training + full interpretability pipeline completed.")
