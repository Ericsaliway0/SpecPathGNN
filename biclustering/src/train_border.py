# =========================
# Standard library
# =========================
import os
from matplotlib.patches import Patch
import math
import json
import itertools
from collections import defaultdict
from matplotlib.gridspec import GridSpec
from itertools import combinations
from tqdm import tqdm
# =========================
# Numerical / scientific
# =========================
from matplotlib.colors import to_rgb
from sklearn.manifold import TSNE
import colorcet as cc
import umap.umap_ as umap
import os
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
import igraph as ig
import leidenalg
from tqdm import tqdm
from matplotlib.ticker import LogFormatterMathtext

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import sem
from matplotlib.colors import LogNorm
# =========================
# Deep learning / graphs
# =========================
import torch
import dgl
import networkx as nx
from matplotlib.ticker import FuncFormatter
# =========================
# Models
# =========================
from .models import (
    GCNModel,
    GINNetModel,
    ChebNetModel,
    MLPPredictor,
    FocalLoss,
)
from matplotlib.colors import PowerNorm

# =========================
# Evaluation utilities
# =========================
from .utils import (
    compute_loss,
    compute_hits_k,
    compute_auc,
    compute_f1,
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_map,
    compute_focalloss,
    compute_focalloss_with_symmetrical_confidence,
    compute_auc_with_symmetrical_confidence,
    compute_f1_with_symmetrical_confidence,
    compute_accuracy_with_symmetrical_confidence,
    compute_precision_with_symmetrical_confidence,
    compute_recall_with_symmetrical_confidence,
    compute_map_with_symmetrical_confidence,
)

# =========================
# Clustering / embeddings
# =========================
from sklearn.cluster import SpectralBiclustering
import umap

# =========================
# Plotting
# =========================
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# =========================
# Survival analysis
# =========================
from lifelines import (
    KaplanMeierFitter,
    CoxPHFitter,
    NelsonAalenFitter,
)
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

from sksurv.util import Surv
from sksurv.metrics import (
    concordance_index_ipcw,
    cumulative_dynamic_auc,
)

# =========================
# Interactive visualization
# =========================
import plotly.graph_objects as go
CLUSTER_COLORS = {
    0: '#0077B6',   1: '#0000FF',   2: '#00B4D8',   3: '#48EAC4',
    4: '#F1C0E8',   5: '#B9FBC0',   6: '#32CD32',   7: '#bee1e6',
    8: '#8A2BE2',   9: '#E377C2',  10: '#8EECF5',  11: '#A3C4F3',
    12: '#FFB347', 13: '#FFD700',  14: '#FF69B4',  15: '#CD5C5C',
    16: '#7FFFD4', 17: '#FF7F50',  18: '#C71585',  19: '#20B2AA', 
    20: "#48CAE4", 21: "#90DBF4",  22: "#0077B6",  23: "#00B4D8"
}

# import os
# import numpy as np
# import pandas as pd
# import torch
# import dgl
# from collections import defaultdict

# from gnn_models import GINNetModel
# from relevance import compute_relevance_scores
# from utils import (
#     plot_gene_umap,
#     plot_joint_gene_pathway_umap,
#     build_saliency_pathway_matrix,
#     run_biclustering,
#     save_top_gene_pathway_pairs,
#     save_cluster_pathway_gene_flows,
#     assign_gene_clusters,
#     align_expression_matrix,
#     extract_gene_cluster_map
# )
import os
# import itertools
# import numpy as np
# import pandas as pd
# import torch
# import dgl
# from collections import defaultdict
# from lifelines import KaplanMeierFitter, CoxPHFitter
# from lifelines.statistics import logrank_test
# from compute_relevance import compute_relevance_scores  # your saliency function
# from gnn_models import GINNetModel, MLPPredictor, FocalLoss
# from plotting_utils import (
#     plot_gene_umap,
#     plot_joint_gene_pathway_umap,
#     plot_patient_cluster_heatmap,
#     plot_patient_bicluster_heatmap,
#     umap_patients,
#     dca_patients,
#     risk_violin_plots,
#     nelson_aalen_plots,
#     plot_km_clusters,
#     plot_sankey_all,
#     plot_cluster_sankey,
#     plot_gene_pathway_modules,
#     gene_pathway_heatmaps,
#     km_pathway_family
# )
# from preprocessing_utils import (
#     load_survival,
#     preprocess_expression,
#     compute_patient_cluster_scores,
#     add_high_low_groups,
#     align_expression_matrix
# )
# from biclustering_utils import (
#     run_biclustering,
#     build_saliency_pathway_matrix,
#     save_top_gene_pathway_pairs,
#     save_cluster_pathway_gene_flows,
#     assign_gene_clusters,
#     extract_gene_cluster_map,
#     map_genes_to_clusters
# )
# from patient_pathway_utils import patient_pathway_family_scores, cox_pathway_family


def train_and_evaluate(args, G_dgl, node_features):

    u, v = G_dgl.edges()
    eids = np.random.permutation(np.arange(G_dgl.number_of_edges()))

    test_size = int(len(eids) * 0.1)
    val_size  = int(len(eids) * 0.1)

    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    val_pos_u,  val_pos_v  = u[eids[test_size:test_size + val_size]], v[eids[test_size:test_size + val_size]]
    train_pos_u, train_pos_v = u[eids[test_size + val_size:]], v[eids[test_size + val_size:]]

    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(G_dgl.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), G_dgl.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    val_neg_u,  val_neg_v  = neg_u[neg_eids[test_size:test_size + val_size]], neg_v[neg_eids[test_size:test_size + val_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size + val_size:]], neg_v[neg_eids[test_size + val_size:]]

    train_g = dgl.remove_edges(G_dgl, eids[:test_size + val_size])

    def create_graph(u, v):
        return dgl.graph((u, v), num_nodes=G_dgl.number_of_nodes())

    train_pos_g = create_graph(train_pos_u, train_pos_v)
    train_neg_g = create_graph(train_neg_u, train_neg_v)

    ########################################
    # 2. MODEL AND OPTIMIZER
    ########################################

    # model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)

    # model = GINNetModel(
    #     graph=G_dgl,
    #     in_feats=node_features.shape[1],
    #     dim_latent=args.dim_latent,
    #     num_layers=args.num_layers,
    #     do_train=True
    # )

    # in_feats = node_features.shape[1]
    # dim_latent = args.dim_latent

    # model = ChebNetModel(
    #     graph=G_dgl,
    #     in_feats=in_feats,
    #     dim_latent=dim_latent,
    #     num_layers=args.num_layers,
    #     k=3,
    #     do_train=True
    # )


    model = GCNModel(
        node_features.shape[1], 
        dim_latent=args.dim_latent, 
        num_layers=args.num_layers, 
        do_train=True
    )

    pred = MLPPredictor(args.input_size, args.hidden_size)

    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    optimizer = torch.optim.Adam(
        itertools.chain(model.parameters(), pred.parameters()),
        lr=args.lr
    )

    output_path = "link_prediction_gcn/results/"
    os.makedirs(output_path, exist_ok=True)

    ########################################
    # 3. TRAINING LOOP
    ########################################

    for e in range(args.epochs):
        model.train()

        h = model(train_g, train_g.ndata["feat"])
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
            print(f"Epoch {e:03d} | Loss: {loss.item():.4f}")

    ########################################
    # 4. NODE-LEVEL RELEVANCE (BEFORE BICLUSTERING)
    ########################################

    relevance_scores = load_or_compute_relevance(
        model=model,
        graph=G_dgl,
        features=G_dgl.ndata["feat"],
        method="saliency",
        cache_dir=os.path.join(output_path, "relevance"),
        force_recompute=False
    )

    # relevance_scores = compute_relevance_scores(
    #     model=model,
    #     graph=G_dgl,
    #     features=G_dgl.ndata["feat"],
    #     node_indices=None,
    #     method="saliency",     # can switch to "integrated_gradients"
    #     use_abs=True,
    #     baseline=None,
    #     steps=50
    # )

    # Collapse feature relevance → gene relevance
    gene_saliency = relevance_scores.sum(dim=1)
    saliency_np = gene_saliency.detach().cpu().numpy()

    ########################################
    # 5. EDGE-LEVEL ATTRIBUTION (IG)
    ########################################


    model.eval()
    with torch.no_grad():
        h_test = model(G_dgl, G_dgl.ndata["feat"])

    u_all, v_all = G_dgl.edges()

    edge_ig = edge_integrated_gradients_cached(
        h=h_test,
        u_idx=u_all,
        v_idx=v_all,
        predictor=pred,
        output_path=output_path,
        steps=50,
        batch_size=4096
    )

    edge_ig_norm = (edge_ig - edge_ig.min()) / (edge_ig.max() - edge_ig.min() + 1e-9)

    pd.DataFrame(
        {
            "Gene_u": u_all.cpu().numpy(),
            "Gene_v": v_all.cpu().numpy(),
            "IG_Score": edge_ig_norm
        }
    ).sort_values(
        "IG_Score", ascending=False
    ).to_csv(
        os.path.join(output_path, "edge_attributions_all.csv"),
        index=False
    )

    df_enrich = pd.read_csv(
        "../gene_pathway_embedding/data/processed/gene_gene_pairs_with_pathwayA_enrichment_network.csv"
    )

    df_enrich = df_enrich[df_enrich["significance"] == "significant"]
    df_enrich["enrich_score"] = -np.log10(df_enrich["pvalue"])

    enrich_matrix, gene_names, pathway_names = load_or_build_enrichment_matrix(
        df_enrich,
        output_path
    )

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    pathway_to_idx = {p: i for i, p in enumerate(pathway_names)}

    saliency_pathway_matrix = np.zeros(
        (len(pathway_names), len(gene_names))
    )

    # for g, gi in tqdm(
    #     gene_to_idx.items(),
    #     total=len(gene_to_idx),
    #     desc="Gene–Pathway saliency"
    # ):
    #     saliency_pathway_matrix[:, gi] = (
    #         saliency_np[gi] * enrich_matrix[gi]
    #     )
    for g, gi in tqdm(
        gene_to_idx.items(),
        total=len(gene_to_idx),
        desc="Gene–Pathway saliency"
    ):
        for p, pi in pathway_to_idx.items():
            saliency_pathway_matrix[pi, gi] = (
                saliency_np[gi] * enrich_matrix[gi, pi]
            )
    np.save(
        os.path.join(output_path, "saliency_pathway_matrix.npy"),
        saliency_pathway_matrix
    )







    bicluster = SpectralBiclustering(
        n_clusters=(10, 10),
        method="log",
        random_state=0
    )

    bicluster.fit(saliency_pathway_matrix)

    row_labels = bicluster.row_labels_
    col_labels = bicluster.column_labels_

    save_top_gene_pathway_pairs(saliency_pathway_matrix, gene_names, pathway_names, output_path)
    save_cluster_pathway_gene_flows(saliency_pathway_matrix, row_labels, gene_names, pathway_names, output_path)
    gene_clusters = assign_gene_clusters(gene_names, col_labels)
    gene_cluster_map = extract_gene_cluster_map(saliency_pathway_matrix, row_labels)
    
    

    # Order indices by cluster
    row_order = np.argsort(row_labels)
    col_order = np.argsort(col_labels)

    saliency_reordered = saliency_pathway_matrix[
        row_order, :
    ][:, col_order]

    pathways_reordered = [pathway_names[i] for i in row_order]
    genes_reordered = [gene_names[i] for i in col_order]

    # Gene names from enrichment matrix
    gene_names = sorted(df_enrich["Gene2"].unique())

    # Assign gene → cluster from biclustering
    gene_clusters = pd.Series(
        col_labels,
        index=gene_names
    )

    print(gene_clusters.value_counts())







    

    pathway_clusters, gene_clusters, leiden_labels = leiden_bipartite_from_saliency(
        saliency_pathway_matrix,
        gene_names,
        pathway_names,
        resolution=1.2,
        weight_threshold=np.percentile(saliency_pathway_matrix, 75)
    )


    leiden_heatmap_file = (
        f"leiden_gene_pathway_heatmap"
        f"_dim{args.dim_latent}"
        f"_lay{args.num_layers}"
        f"_hid{args.hidden_size}"
        f"_epo{args.epochs}.png"
    )

    # plot_leiden_saliency_heatmap(
    #     saliency_pathway_matrix=saliency_pathway_matrix,
    #     gene_names=gene_names,
    #     pathway_names=pathway_names,
    #     gene_clusters=gene_clusters,
    #     pathway_clusters=pathway_clusters,
    #     output_path=output_path,
    #     filename=leiden_heatmap_file
    # )
    # plot_leiden_saliency_heatmap(
    #     saliency_pathway_matrix=saliency_pathway_matrix,
    #     gene_names=gene_names,
    #     pathway_names=pathway_names,
    #     gene_clusters=gene_clusters,
    #     pathway_clusters=pathway_clusters,
    #     output_path=output_path,
    #     filename=leiden_heatmap_file
    # )
    plot_leiden_saliency_heatmap(
        saliency_pathway_matrix=saliency_pathway_matrix,
        gene_names=gene_names,
        pathway_names=pathway_names,
        gene_clusters=gene_clusters,
        pathway_clusters=pathway_clusters,
        output_path=output_path
    )
    
    ############################################################################################################
            
        
    ########################################
    # TYPE-B SANKEY:
    # Gene Cluster → Pathway Cluster → Family
    ########################################
    

    PATHWAY_FAMILY_MAP = {
        "PI3K-AKT signaling pathway": "PI3K/AKT",
        "mTOR signaling pathway": "PI3K/AKT",
        "MAPK signaling pathway": "MAPK",
        "ERK cascade": "MAPK",
        "p53 signaling pathway": "Cell Cycle / DNA Damage",
        "Cell cycle": "Cell Cycle / DNA Damage",
        "DNA repair": "Cell Cycle / DNA Damage",
    }

    def map_to_family(p):
        return PATHWAY_FAMILY_MAP.get(p, "Other")    


    # ==========================================================
    # 1. BUILD CLUSTER → COLOR MAPS (MATCH LEIDEN HEATMAP)
    # ==========================================================

    def make_cluster_cmap_from_labels(cluster_ids):
        uniq = []
        for c in cluster_ids:
            if c not in uniq:
                uniq.append(c)
        palette = cc.glasbey[:len(uniq)]
        return dict(zip(uniq, palette))


    gene_cluster_ids = sorted(set(gene_clusters.values()))
    pathway_cluster_ids = sorted(set(pathway_clusters.values()))

    gene_cmap = make_cluster_cmap_from_labels(gene_cluster_ids)
    pathway_cmap = make_cluster_cmap_from_labels(pathway_cluster_ids)


    # ==========================================================
    # 2. FAMILY COLORS (SEMANTIC & STABLE)
    # ==========================================================

    FAMILY_COLORS = {
        "PI3K/AKT": "#55A868",
        "MAPK": "#C44E52",
        "Cell Cycle / DNA Damage": "#8172B2",
        "Immune": "#64B5CD",
        "Metabolism": "#CCB974",
        "Other": "#999999",
    }


    # ==========================================================
    # 3. BUILD LONG-FORM FLOW TABLE
    # ==========================================================

    rows = []

    for gi, gene in enumerate(gene_names):
        g_cluster = f"G{gene_clusters[gene]}"

        for pi, pathway in enumerate(pathway_names):
            score = saliency_pathway_matrix[pi, gi]
            if score <= 0:
                continue

            p_cluster = f"P{pathway_clusters[pathway]}"
            family = PATHWAY_FAMILY_MAP.get(pathway, "Other")

            rows.append({
                "gene_cluster": g_cluster,
                "pathway_cluster": p_cluster,
                "family": family,
                "weight": score,
            })

    df = pd.DataFrame(rows)


    # ==========================================================
    # 4. AGGREGATE FLOWS
    # ==========================================================

    df_gp = (
        df.groupby(["gene_cluster", "pathway_cluster"])["weight"]
        .sum()
        .reset_index()
    )

    df_pf = (
        df.groupby(["pathway_cluster", "family"])["weight"]
        .sum()
        .reset_index()
    )

    ########################################
    # TOP-K FLOW FILTERING (PER CLUSTER)
    ########################################

    TOP_K_GENE_TO_PATHWAY = 6     # ← tune (5–10 works well)
    TOP_K_PATHWAY_TO_FAMILY = 5  # ← tune


    # --- Gene cluster → Pathway cluster ---
    df_gp_topk = (
        df_gp
        .sort_values("weight", ascending=False)
        .groupby("gene_cluster", group_keys=False)
        .head(TOP_K_GENE_TO_PATHWAY)
    )


    # --- Pathway cluster → Family ---
    df_pf_topk = (
        df_pf
        .sort_values("weight", ascending=False)
        .groupby("pathway_cluster", group_keys=False)
        .head(TOP_K_PATHWAY_TO_FAMILY)
    )


    # ==========================================================
    # 5. BUILD SANKEY EDGE LIST
    # ==========================================================

    ########################################
    # BUILD SANKEY EDGES (TOP-K FILTERED)
    ########################################

    edges = []

    for _, r in df_gp_topk.iterrows():
        edges.append({
            "source": r["gene_cluster"],
            "target": r["pathway_cluster"],
            "value": r["weight"],
        })

    for _, r in df_pf_topk.iterrows():
        edges.append({
            "source": r["pathway_cluster"],
            "target": r["family"],
            "value": r["weight"],
        })

    df_sankey = pd.DataFrame(edges)


    MIN_FLOW_WEIGHT = np.percentile(df_sankey["value"], 70)
    df_sankey = df_sankey[df_sankey["value"] >= MIN_FLOW_WEIGHT]

    # ==========================================================
    # 6. NODE INDEXING
    # ==========================================================

    labels = pd.unique(df_sankey[["source", "target"]].values.ravel())
    label_to_id = {l: i for i, l in enumerate(labels)}


    # ==========================================================
    # 7. NODE COLOR FUNCTION (MATCH HEATMAP)
    # ==========================================================

    def node_color(label):
        if label.startswith("G"):
            cid = int(label[1:])
            return gene_cmap.get(cid, "#333333")

        if label.startswith("P"):
            cid = int(label[1:])
            return pathway_cmap.get(cid, "#666666")

        return FAMILY_COLORS.get(label, "#BBBBBB")


    node_colors = [node_color(l) for l in labels]


    # ==========================================================
    # 8. CREATE SANKEY FIGURE
    # ==========================================================

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=labels,
            color=node_colors,
            pad=14,
            thickness=16,
        ),
        link=dict(
            source=df_sankey["source"].map(label_to_id),
            target=df_sankey["target"].map(label_to_id),
            value=df_sankey["value"],
        )
    ))

    fig.update_layout(
        title="Gene Cluster → Pathway Cluster → Pathway Family",
        font_size=12,
    )


    # ==========================================================
    # 9. SAVE OUTPUTS (HTML + PNG)
    # ==========================================================

    # 

    fig.write_html(
        os.path.join(output_path, "sankey_gene_pathway_family.html")
    )

    # PNG export (requires: pip install kaleido)
    fig.write_image(
        os.path.join(output_path, "sankey_gene_pathway_family.png"),
        width=1600,
        height=900,
        scale=2,
    )

    print("✅ Sankey saved (HTML + PNG)")


    ########################################
    # 2. LOAD & ALIGN EXPRESSION DATA
    ########################################


    expr_df = pd.read_csv(
        "../ACGNN/data/TCGA-BRCA.expression.tsv",
        sep="\t",
        index_col=0
    )
    
    expr_genes = set(expr_df.index)
    gnn_genes = set(gene_names)
    common_genes = sorted(expr_genes.intersection(gnn_genes))

    expr_df = expr_df.loc[common_genes]

    gene_to_expr_idx = {g: i for i, g in enumerate(common_genes)}
    
    bicluster_gene_ids = genes_reordered
    bicluster_pathway_ids = pathways_reordered
    valid_genes = expr_df.index.intersection(bicluster_gene_ids)

    expr_matrix = expr_df.loc[valid_genes].T.to_numpy()

    np.save(
        os.path.join(output_path, "bicluster_row_labels.npy"),
        bicluster.row_labels_
    )


    ########################################
    # 12. GENE–PATHWAY EXPLANATION TABLE
    ########################################

    rows = []
    for p in range(saliency_pathway_matrix.shape[0]):
        top_genes = np.argsort(
            saliency_pathway_matrix[p]
        )[::-1][:20]

        for g in top_genes:
            rows.append([
                pathway_names[p],
                gene_names[g],
                saliency_pathway_matrix[p, g]
            ])




    # -----------------------------------------
    # Select ≤50 genes for gene-level Sankey
    # -----------------------------------------
    MAX_GENES = 100

    # Total saliency per gene
    gene_scores = saliency_pathway_matrix.sum(axis=0)

    # Top genes by saliency
    top_gene_idx = np.argsort(gene_scores)[::-1][:MAX_GENES]

    selected_genes = [gene_names[i] for i in top_gene_idx]

    rows = []

    for g in selected_genes:
        gi = gene_names.index(g)

        for pi, pathway in enumerate(pathway_names):
            score = saliency_pathway_matrix[pi, gi]
            if score <= 0:
                continue

            rows.append({
                "gene": g,
                "pathway": pathway,
                "weight": score,
            })

    df = pd.DataFrame(rows)



    TOP_K_PATHWAYS_PER_GENE = 3
    TOP_K_GENE_TO_PATHWAY = 3 


    df_gp = (
        df.groupby(["gene", "pathway"])["weight"]
        .sum()
        .reset_index()
    )

    df_gp_topk = (
        df_gp
        .sort_values("weight", ascending=False)
        .groupby("gene", group_keys=False)
        .head(TOP_K_PATHWAYS_PER_GENE)
    )

    MIN_FLOW = np.percentile(df_gp_topk["weight"], 60)
    df_gp_topk = df_gp_topk[df_gp_topk["weight"] >= MIN_FLOW]

    df_sankey = df_gp_topk.rename(
        columns={"gene": "source", "pathway": "target"}
    )

    # ----------------------------------------------------
    # Nodes
    # ----------------------------------------------------
    labels = pd.unique(df_sankey[["source", "target"]].values.ravel())
    label_to_id = {l: i for i, l in enumerate(labels)}

    # ----------------------------------------------------
    # Fixed cluster-based node colors (CANONICAL)
    # ----------------------------------------------------
    def node_color(label):
        # Gene node
        if label in gene_clusters:
            cid = gene_clusters[label]
            return CLUSTER_COLORS[cid]

        # Pathway node
        if label in pathway_clusters:
            cid = pathway_clusters[label]
            return CLUSTER_COLORS[cid]

        # Fallback (should not happen)
        return "#BDBDBD"

    node_colors = [node_color(l) for l in labels]

    # ----------------------------------------------------
    # Sankey plot
    # ----------------------------------------------------
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=labels,
            color=node_colors,
            pad=12,
            thickness=14,
            line=dict(color="black", width=0.5),
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=df_sankey["source"].map(label_to_id),
            target=df_sankey["target"].map(label_to_id),
            value=df_sankey["weight"],
        ),
    ))

    fig.update_layout(
        font_size=18,
        width=600,
        height=1800,
    )

    # ----------------------------------------------------
    # Save
    # ----------------------------------------------------
    os.makedirs(output_path, exist_ok=True)

    fig.write_html(
        os.path.join(output_path, "sankey_gene_to_pathway.html")
    )

    fig.write_image(
        os.path.join(output_path, "sankey_gene_to_pathway.png"),
        width=600,
        height=1800,
        scale=2,
    )





    ############################################################################################################
    

    # Sort clusters
    sorted_pathway_clusters = sort_pathway_clusters_by_saliency(
        saliency_pathway_matrix, pathway_names, pathway_clusters
    )

    sorted_gene_clusters = sort_gene_clusters_by_saliency(
        saliency_pathway_matrix, gene_names, gene_clusters
    )

    # Sort members inside clusters
    pathway_order = sort_pathways_within_clusters(
        saliency_pathway_matrix,
        pathway_names,
        pathway_clusters,
        sorted_pathway_clusters,
    )

    gene_order = sort_genes_within_clusters(
        saliency_pathway_matrix,
        gene_names,
        gene_clusters,
        sorted_gene_clusters,
    )

    # Sorted matrix
    saliency_sorted = saliency_pathway_matrix[np.ix_(pathway_order, gene_order)]

    # Sorted labels
    pathway_names_sorted = [pathway_names[i] for i in pathway_order]
    gene_names_sorted = [gene_names[i] for i in gene_order]




    # ==============================================================
    # Parameters
    # ==============================================================

    MIN_SALIENCY = 1e-2          # 🔑 skip low saliency FIRST
    MAX_GENES_PER_CLUSTER = 3
    MAX_PATHWAYS_PER_CLUSTER = 3

    # ==============================================================
    # Sort clusters & members (REQUIRED)
    # ==============================================================

    sorted_pathway_clusters = sort_pathway_clusters_by_saliency(
        saliency_pathway_matrix, pathway_names, pathway_clusters
    )

    sorted_gene_clusters = sort_gene_clusters_by_saliency(
        saliency_pathway_matrix, gene_names, gene_clusters
    )

    pathway_order = sort_pathways_within_clusters(
        saliency_pathway_matrix,
        pathway_names,
        pathway_clusters,
        sorted_pathway_clusters,
    )

    gene_order = sort_genes_within_clusters(
        saliency_pathway_matrix,
        gene_names,
        gene_clusters,
        sorted_gene_clusters,
    )

    saliency_sorted = saliency_pathway_matrix[np.ix_(pathway_order, gene_order)]
    pathway_names_sorted = [pathway_names[i] for i in pathway_order]
    gene_names_sorted    = [gene_names[i] for i in gene_order]

    # ==============================================================
    # Build edge list (skip LOW saliency FIRST) ✅
    # ==============================================================

    rows = []

    for pi, p in enumerate(pathway_names_sorted):
        for gi, g in enumerate(gene_names_sorted):
            w = saliency_sorted[pi, gi]

            if w <= MIN_SALIENCY:
                continue

            rows.append({
                "gene": g,
                "pathway": p,
                "weight": float(w),
                "gene_cluster": f"G{gene_clusters[g]}",
                "pathway_cluster": f"P{pathway_clusters[p]}",
            })

    df = pd.DataFrame(rows)

    # ==============================================================
    # Aggregate saliency scores
    # ==============================================================

    gene_score = (
        df.groupby(["gene", "gene_cluster"])["weight"]
        .sum()
        .reset_index()
    )

    pathway_score = (
        df.groupby(["pathway", "pathway_cluster"])["weight"]
        .sum()
        .reset_index()
    )

    # ==============================================================
    # Prune: TOP genes per gene cluster
    # ==============================================================

    top_genes = (
        gene_score
        .sort_values("weight", ascending=False)
        .groupby("gene_cluster")
        .head(MAX_GENES_PER_CLUSTER)["gene"]
        .unique()
    )

    df = df[df["gene"].isin(top_genes)]

    # ==============================================================
    # Prune: TOP pathways per pathway cluster
    # ==============================================================

    top_pathways = (
        pathway_score
        .sort_values("weight", ascending=False)
        .groupby("pathway_cluster")
        .head(MAX_PATHWAYS_PER_CLUSTER)["pathway"]
        .unique()
    )

    df = df[df["pathway"].isin(top_pathways)]

    # ==============================================================
    # Final labels & Sankey mapping
    # ==============================================================

    labels = pd.unique(df[["gene", "pathway"]].values.ravel())
    label_to_id = {l: i for i, l in enumerate(labels)}

    sources = df["gene"].map(label_to_id)
    targets = df["pathway"].map(label_to_id)
    values  = df["weight"]

    # ==============================================================
    # Node colors
    # ==============================================================

    def node_color(label):
        if label in gene_clusters:
            return gene_cmap[gene_clusters[label]]
        if label in pathway_clusters:
            return pathway_cmap[pathway_clusters[label]]
        return "#BBBBBB"

    node_colors = [node_color(l) for l in labels]

    # ==============================================================
    # Sankey plot
    # ==============================================================

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=labels,
            color=node_colors,
            pad=14,
            thickness=16,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
        ),
    ))

    fig.update_layout(
        # title="Gene → Pathway Saliency Flow (Bidirectionally Pruned)",
        font_size=18,
    )

    fig.write_html(os.path.join(output_path, "gene_pathway_sankey_pruned.html"))
    fig.write_image(
        os.path.join(output_path, "gene_pathway_sankey_pruned.png"),
        width=600,
        height=1800,
        scale=2,
    )

    print("✅ Bidirectionally pruned Sankey saved")

    #######################################################################################################
    
    rows = []

    for pi, p in enumerate(pathway_names_sorted):
        for gi, g in enumerate(gene_names_sorted):
            w = saliency_sorted[pi, gi]
            if w <= 0:
                continue

            rows.append({
                "gene": g,
                "pathway": p,
                "weight": w,
                "gene_cluster": f"G{gene_clusters[g]}",
                "pathway_cluster": f"P{pathway_clusters[p]}",
            })

    df = pd.DataFrame(rows)


    TOP_GENES = 50

    top_genes = (
        df.groupby("gene")["weight"]
        .sum()
        .sort_values(ascending=False)
        .head(TOP_GENES)
        .index
    )

    df = df[df["gene"].isin(top_genes)]


    labels = pd.unique(df[["gene", "pathway"]].values.ravel())
    label_to_id = {l: i for i, l in enumerate(labels)}

    sources = df["gene"].map(label_to_id)
    targets = df["pathway"].map(label_to_id)
    values  = df["weight"]


    def node_color(label):
        if label in gene_clusters:
            return gene_cmap[gene_clusters[label]]
        if label in pathway_clusters:
            return pathway_cmap[pathway_clusters[label]]
        return "#BBBBBB"

    node_colors = [node_color(l) for l in labels]


    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=labels,
            color=node_colors,
            pad=14,
            thickness=16,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
        ),
    ))

    fig.update_layout(
        # title="Gene → Pathway Saliency Flow",
        font_size=18,
    )

    # 

    fig.write_html(os.path.join(output_path, "gene_pathway_sankey_sorted.html"))
    fig.write_image(
        os.path.join(output_path, "gene_pathway_sankey_sorted.png"),
        width=600,
        height=1800,
        scale=2,
    )

    print("✅ Sorted gene → pathway Sankey saved")


    ############################################################################################################
    
    rows = []

    for gi, gene in enumerate(gene_names):
        g_cluster = f"G{gene_clusters[gene]}"

        for pi, pathway in enumerate(pathway_names):
            score = saliency_pathway_matrix[pi, gi]
            if score <= 0:
                continue

            p_cluster = f"P{pathway_clusters[pathway]}"

            rows.append({
                "gene_cluster": g_cluster,
                "pathway_cluster": p_cluster,
                "weight": score,
            })

    df = pd.DataFrame(rows)

    df_cluster = (
        df.groupby(["gene_cluster", "pathway_cluster"])["weight"]
        .sum()
        .reset_index()
    )

    MIN_WEIGHT = np.percentile(df_cluster["weight"], 60)
    df_cluster = df_cluster[df_cluster["weight"] >= MIN_WEIGHT]


    def make_cluster_cmap(cluster_ids):
        uniq = sorted(cluster_ids)
        palette = cc.glasbey[:len(uniq)]
        return dict(zip(uniq, palette))

    gene_cluster_ids = sorted(set(gene_clusters.values()))
    pathway_cluster_ids = sorted(set(pathway_clusters.values()))

    gene_cmap = make_cluster_cmap(gene_cluster_ids)
    pathway_cmap = make_cluster_cmap(pathway_cluster_ids)


    gene_strength = (
        df_cluster.groupby("gene_cluster")["weight"].sum().to_dict()
    )

    pathway_strength = (
        df_cluster.groupby("pathway_cluster")["weight"].sum().to_dict()
    )

    gene_order = sorted(
        gene_strength.keys(),
        key=lambda g: -gene_strength[g]
    )

    pathway_order = sorted(
        pathway_strength.keys(),
        key=lambda p: -pathway_strength[p]
    )

    labels = gene_order + pathway_order
    label_to_id = {l: i for i, l in enumerate(labels)}

    node_colors = []

    for l in labels:
        if l.startswith("G"):
            node_colors.append(gene_cmap[int(l[1:])])
        else:
            node_colors.append(pathway_cmap[int(l[1:])])

    sources = df_cluster["gene_cluster"].map(label_to_id)
    targets = df_cluster["pathway_cluster"].map(label_to_id)
    values  = df_cluster["weight"]

    v = values.to_numpy()
    alpha = 0.25 + 0.75 * (v - v.min()) / (v.max() - v.min())

    link_colors = [f"rgba(0,0,0,{a:.2f})" for a in alpha]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=labels,
            color=node_colors,
            pad=20,
            thickness=18,
            line=dict(color="black", width=0.5),
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        ),
    ))

    fig.update_layout(
        title="Gene Cluster → Pathway Cluster Saliency Flow",
        font_size=18,
        width=600,
        height=1200,
    )

    # output_path = "results/sankey"
    # 

    fig.write_html(os.path.join(output_path, "cluster_sankey.html"))
    fig.write_image(
        os.path.join(output_path, "cluster_sankey.png"),
        width=600,
        height=1200,
        scale=2,
    )


    ############################################################################################################

    # ----------------------------------------------------
    # Build Gene → Pathway cluster saliency table
    # ----------------------------------------------------
    rows = []

    for gi, gene in enumerate(gene_names):
        g_cluster = f"G{gene_clusters[gene]}"

        for pi, pathway in enumerate(pathway_names):
            score = saliency_pathway_matrix[pi, gi]
            if score <= 0:
                continue

            p_cluster = f"P{pathway_clusters[pathway]}"

            rows.append({
                "gene_cluster": g_cluster,
                "pathway_cluster": p_cluster,
                "weight": score,
            })

    df = pd.DataFrame(rows)

    # ----------------------------------------------------
    # Aggregate cluster-to-cluster saliency
    # ----------------------------------------------------
    df_gp = (
        df.groupby(["gene_cluster", "pathway_cluster"])["weight"]
        .sum()
        .reset_index()
    )

    # ----------------------------------------------------
    # Threshold weak connections
    # ----------------------------------------------------
    MIN_WEIGHT = np.percentile(df_gp["weight"], 60)
    df_gp = df_gp[df_gp["weight"] >= MIN_WEIGHT]

    # ----------------------------------------------------
    # Compute strengths for ordering
    # ----------------------------------------------------
    gene_strength = (
        df_gp.groupby("gene_cluster")["weight"].sum().to_dict()
    )

    pathway_strength = (
        df_gp.groupby("pathway_cluster")["weight"].sum().to_dict()
    )

    gene_order = sorted(gene_strength, key=lambda x: -gene_strength[x])
    pathway_order = sorted(pathway_strength, key=lambda x: -pathway_strength[x])

    # ----------------------------------------------------
    # Color maps
    def cluster_color_from_label(label):
        """
        label: 'G3' or 'P7'
        """
        cid = int(label[1:])  # strip 'G' or 'P'
        if cid not in CLUSTER_COLORS:
            raise ValueError(f"Cluster ID {cid} not in CLUSTER_COLORS")
        return CLUSTER_COLORS[cid]

    gene_cmap = {g: cluster_color_from_label(g) for g in gene_order}
    pathway_cmap = {p: cluster_color_from_label(p) for p in pathway_order}

    # ----------------------------------------------------
    # Sankey node layout
    # ----------------------------------------------------
    labels = gene_order + pathway_order
    label_to_id = {l: i for i, l in enumerate(labels)}

    def y_positions(order):
        n = len(order)
        if n == 1:
            return [0.5]
        return np.linspace(0.05, 0.95, n)

    gene_y = y_positions(gene_order)
    pathway_y = y_positions(pathway_order)

    x_pos, y_pos, node_colors = [], [], []

    for l in labels:
        if l.startswith("G"):
            x_pos.append(0.05)
            y_pos.append(gene_y[gene_order.index(l)])
            node_colors.append(gene_cmap[l])
        else:
            x_pos.append(0.95)
            y_pos.append(pathway_y[pathway_order.index(l)])
            node_colors.append(pathway_cmap[l])

    # ----------------------------------------------------
    # Sankey links
    # ----------------------------------------------------
    sources = df_gp["gene_cluster"].map(label_to_id)
    targets = df_gp["pathway_cluster"].map(label_to_id)
    values  = df_gp["weight"].to_numpy()

    # # Transparency scaled by saliency
    # alpha = 0.25 + 0.75 * (values - values.min()) / (values.max() - values.min())
    # link_colors = [f"rgba(0,0,0,{a:.2f})" for a in alpha]
    def blend_colors(hex1, hex2, alpha):
        def hex_to_rgb(h):
            h = h.lstrip("#")
            return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)])

        rgb = 0.5 * hex_to_rgb(hex1) + 0.5 * hex_to_rgb(hex2)
        r, g, b = rgb.astype(int)
        return f"rgba({r},{g},{b},{alpha:.2f})"

    link_colors = [
        blend_colors(
            gene_cmap[row["gene_cluster"]],
            pathway_cmap[row["pathway_cluster"]],
            0.25 + 0.75 * (row["weight"] - values.min()) / (values.max() - values.min())
        )
        for _, row in df_gp.iterrows()
    ]

    # ----------------------------------------------------
    # Plot
    # ----------------------------------------------------
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=labels,
            color=node_colors,
            pad=22,
            thickness=18,
            x=x_pos,
            y=y_pos,
            line=dict(color="black", width=0.5),
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        ),
    ))

    fig.update_layout(
        # title="Gene Cluster → Pathway Cluster Saliency Flow",
        font_size=18,
        width=600,
        height=1800,
    )

    # ----------------------------------------------------
    # Save
    # ----------------------------------------------------
    os.makedirs(output_path, exist_ok=True)

    fig.write_html(os.path.join(output_path, "gene_pathway_sankey.html"))
    fig.write_image(
        os.path.join(output_path, "gene_pathway_sankey.png"),
        width=600,
        height=1800,
        scale=2,
    )

    ############################################################################################################
    














    # edges = []

    # # Gene → Pathway
    # for _, r in df_gp_topk.iterrows():
    #     edges.append({
    #         "source": r["gene"],
    #         "target": r["pathway"],
    #         "value": r["weight"],
    #     })

    # # Pathway → Family
    # for _, r in df_pf_topk.iterrows():
    #     edges.append({
    #         "source": r["pathway"],
    #         "target": r["family"],
    #         "value": r["weight"],
    #     })

    # df_sankey = pd.DataFrame(edges)



    # MIN_FLOW = np.percentile(df_sankey["value"], 60)
    # df_sankey = df_sankey[df_sankey["value"] >= MIN_FLOW]


    # labels = pd.unique(df_sankey[["source", "target"]].values.ravel())
    # label_to_id = {l: i for i, l in enumerate(labels)}


    # def node_color(label):
    #     # Gene
    #     if label in gene_clusters:
    #         return gene_cmap[gene_clusters[label]]

    #     # Pathway
    #     if label in pathway_clusters:
    #         return pathway_cmap[pathway_clusters[label]]

    #     # Family
    #     return FAMILY_COLORS.get(label, "#BBBBBB")

    # node_colors = [node_color(l) for l in labels]



    # fig = go.Figure(go.Sankey(
    #     arrangement="snap",
    #     node=dict(
    #         label=labels,
    #         color=node_colors,
    #         pad=12,
    #         thickness=14,
    #         hovertemplate="%{label}<extra></extra>",
    #     ),
    #     link=dict(
    #         source=df_sankey["source"].map(label_to_id),
    #         target=df_sankey["target"].map(label_to_id),
    #         value=df_sankey["value"],
    #     )
    # ))

    # fig.update_layout(
    #     title="Gene → Pathway → Pathway Family (Gene-level Explanation)",
    #     font_size=11,
    # )



    # 

    # fig.write_html(
    #     os.path.join(output_path, "sankey_gene_level.html")
    # )

    # fig.write_image(
    #     os.path.join(output_path, "sankey_gene_level.png"),
    #     width=1800,
    #     height=1000,
    #     scale=2,
    # )

    # print("✅ Gene-level Sankey saved")




    ########################################
    # 13. BUILD CLUSTER → PATHWAY → GENE FLOWS
    ########################################

    top_k_genes = 10
    flows = []

    for p_idx, cluster_id in enumerate(bicluster.row_labels_):
        pathway = pathway_names[p_idx]
        gene_scores = saliency_pathway_matrix[p_idx]

        top_gene_indices = np.argsort(
            gene_scores
        )[::-1][:top_k_genes]

        for g_idx in top_gene_indices:
            score = gene_scores[g_idx]
            if score <= 0:
                continue

            flows.append([
                f"Cluster {cluster_id}",
                pathway,
                gene_names[g_idx],
                score
            ])

    df_flows = pd.DataFrame(
        flows,
        columns=["Cluster", "Pathway", "Gene", "Value"]
    )

    df_flows.to_csv(
        os.path.join(output_path, "sankey_cluster_pathway_gene.csv"),
        index=False
    )

    gene_cluster_map = defaultdict(list)

    for p_idx, cluster_id in enumerate(bicluster.row_labels_):
        gene_scores = saliency_pathway_matrix[p_idx]
        top_genes = np.argsort(gene_scores)[::-1][:20]

        for g in top_genes:
            gene_cluster_map[cluster_id].append(g)

    # deduplicate
    for c in gene_cluster_map:
        gene_cluster_map[c] = list(set(gene_cluster_map[c]))



    surv = pd.read_csv(
        "../ACGNN/data/TCGA-BRCA.survival.tsv",
        sep="\t"
    )

    if "_PATIENT" in surv.columns:
        surv = surv.rename(columns={"_PATIENT": "patient_id"})
    if "OS.time" in surv.columns:
        surv = surv.rename(columns={"OS.time": "time"})
    if "OS" in surv.columns:
        surv = surv.rename(columns={"OS": "event"})

    surv = surv.set_index("patient_id")
    surv = surv[["time", "event"]]

    surv["time"] = surv["time"].astype(float)
    surv["event"] = surv["event"].astype(int)

    print(f"✅ Survival loaded: {surv.shape[0]} patients")


    expr_df.columns = [c[:12] for c in expr_df.columns]

    # Aggregate multiple samples per patient (mean TPM)
    expr_df = expr_df.groupby(expr_df.columns, axis=1).mean()

    print(f"🧬 Expression patients after collapsing: {expr_df.shape[1]}")

    patient_ids = sorted(
        set(expr_df.columns).intersection(surv.index)
    )

    expr_df = expr_df[patient_ids]
    surv = surv.loc[patient_ids]

    print(f"🧑 Patients used (final): {len(patient_ids)}")

    expr_genes = set(expr_df.index)
    gnn_genes = set(gene_names)

    common_genes = sorted(expr_genes.intersection(gnn_genes))

    print(f"🧬 Common genes: {len(common_genes)}")

    expr_df = expr_df.loc[common_genes]
    

    gene_to_expr_idx = {
        g: i for i, g in enumerate(common_genes)
    }

    expr_matrix = expr_df.T.to_numpy()

    print("expr_df shape:", expr_df.shape)
    print("expr_matrix shape:", expr_matrix.shape)

    assert expr_matrix.shape == (len(patient_ids), len(common_genes))


    gene_cluster_map = defaultdict(set)

    for p_idx, cluster_id in enumerate(bicluster.row_labels_):
        gene_scores = saliency_pathway_matrix[p_idx]
        top_genes = np.argsort(gene_scores)[::-1][:20]

        for g_idx in top_genes:
            gene = gene_names[g_idx]
            if gene in gene_to_expr_idx:
                gene_cluster_map[cluster_id].add(
                    gene_to_expr_idx[gene]
                )

    # convert sets → lists
    gene_cluster_map = {
        c: list(idxs) for c, idxs in gene_cluster_map.items()
    }


    
    rows = []


    patient_pathway_scores = {}

    for p_idx, pathway in enumerate(pathway_names):
        gene_scores = saliency_pathway_matrix[p_idx]
        top_gene_idxs = np.argsort(gene_scores)[::-1][:20]

        expr_gene_idxs = []

        for g_idx in top_gene_idxs:
            gene = gene_names[g_idx]
            if gene in gene_to_expr_idx:
                expr_gene_idxs.append(gene_to_expr_idx[gene])

        # skip empty pathways
        if len(expr_gene_idxs) == 0:
            continue

        patient_pathway_scores[pathway] = (
            expr_matrix[:, expr_gene_idxs].mean(axis=1)
        )

        patient_pathway_scores = pd.DataFrame(
            patient_pathway_scores,
            index=patient_ids
        )



    ########################################
    # 22. PATIENT × PATHWAY-FAMILY SCORES
    ########################################

    # Map each pathway to a family
    pathway_to_family = {
        p: map_to_family(p)
        for p in patient_pathway_scores.columns
    }

    # Build patient × family matrix
    patient_family_scores = {}

    for family in sorted(set(pathway_to_family.values())):
        family_pathways = [
            p for p, f in pathway_to_family.items() if f == family
        ]

        # mean across pathways in the family
        patient_family_scores[family] = (
            patient_pathway_scores[family_pathways].mean(axis=1)
        )

    patient_family_scores = pd.DataFrame(
        patient_family_scores,
        index=patient_ids
    )

    print("🧬 Pathway-family matrix shape:", patient_family_scores.shape)


    ########################################
    # 23. MERGE WITH SURVIVAL
    ########################################

    df_family_surv = surv.join(
        patient_family_scores,
        how="inner"
    )

    print("🧑 Patients for Cox:", df_family_surv.shape[0])



    ########################################
    # 24. COX REGRESSION (PATHWAY FAMILIES)
    ########################################


    cox_family_results = []

    for family in patient_family_scores.columns:
        df_tmp = df_family_surv[["time", "event", family]].dropna()

        # skip degenerate cases
        if df_tmp[family].std() == 0:
            continue

        cph = CoxPHFitter()
        cph.fit(
            df_tmp,
            duration_col="time",
            event_col="event"
        )

        s = cph.summary.loc[family]

        cox_family_results.append({
            "Family": family,
            "HR": s["exp(coef)"],
            "CI_lower": s["exp(coef) lower 95%"],
            "CI_upper": s["exp(coef) upper 95%"],
            "p": s["p"]
        })

    df_cox_family = pd.DataFrame(cox_family_results)


    ########################################
    # 25. FDR CORRECTION
    ########################################

    from statsmodels.stats.multitest import multipletests

    df_cox_family["FDR"] = multipletests(
        df_cox_family["p"],
        method="fdr_bh"
    )[1]

    df_cox_family = df_cox_family.sort_values("p")

    df_cox_family.to_csv(
        os.path.join(output_path, "cox_pathway_families.csv"),
        index=False
    )

    print("✅ Pathway-family Cox results saved.")

    for pathway in patient_pathway_scores.columns:
        fam = map_to_family(pathway)

        # High / Low based on median
        grp = (
            patient_pathway_scores[pathway] >=
            patient_pathway_scores[pathway].median()
        ).map({True: "High", False: "Low"})

        df_tmp = pd.DataFrame({
            "family": fam,
            "risk": grp,
        })

        counts = df_tmp.value_counts().reset_index(name="value")

        for _, r in counts.iterrows():
            rows.append({
                "source": pathway,
                "target": fam,
                "value": r["value"]
            })
            rows.append({
                "source": fam,
                "target": r["risk"],
                "value": r["value"]
            })

    df_sankey = pd.DataFrame(rows)


    labels = pd.unique(df_sankey[["source", "target"]].values.ravel())
    label_to_id = {l: i for i, l in enumerate(labels)}

    fig = go.Figure(go.Sankey(
        node=dict(
            label=labels,
            pad=15,
            thickness=15,
        ),
        link=dict(
            source=df_sankey["source"].map(label_to_id),
            target=df_sankey["target"].map(label_to_id),
            value=df_sankey["value"],
        )
    ))

    fig.update_layout(
        title="Pathway → Family → Survival Risk Sankey",
        font_size=12,
    )

    fig.write_html(
        os.path.join(output_path, "sankey_pathway_family_survival.html")
    )
        
    
    
    
    
    
    
    

    # plot_leiden_saliency_heatmap(
    #     saliency_pathway_matrix,
    #     gene_names,
    #     pathway_names,
    #     gene_clusters,
    #     pathway_clusters,
    #     output_path,
    #     sort_by_cluster=True
    # )


    gene_saliency = saliency_pathway_matrix.T  # (genes × pathways)
    gene_cluster_ids = [gene_clusters[g] for g in gene_names]


    gene_umap_file = os.path.join(
        output_path,
        f"gene_umap_leiden"
        f"_dim{args.dim_latent}"
        f"_lay{args.num_layers}"
        f"_epo{args.epochs}.png"
    )

    plot_umap_with_cluster_colors(
        X=gene_saliency,
        cluster_ids=gene_cluster_ids,
        cluster_colors=CLUSTER_COLORS,
        output_path=output_path,
        filename=os.path.basename(gene_umap_file),
        title="Gene Saliency UMAP",
    )

    # embedding_tsne = plot_tsne_with_cluster_colors(
    #     X=gene_saliency,
    #     cluster_ids=gene_cluster_ids,
    #     cluster_colors=CLUSTER_COLORS,
    #     output_path=output_path,
    #     filename="gene_tsne_leiden_tuning.png",
    #     title="t-SNE of Gene Saliency (Leiden)",
    #     perplexity=30,        # 20–50 typical
    #     n_iter=2000,          # smoother convergence
    #     metric="cosine",      # important for saliency vectors
    #     random_state=42,
    # )


    tsne_gene_file = os.path.join(
        output_path,
        f"gene_tsne_leiden"
        f"_dim{args.dim_latent}"
        f"_lay{args.num_layers}"
        f"_hid{args.hidden_size}"
        f"_epo{args.epochs}.png"
    )
    
    # tsne_gene_file = os.path.join(
    #     output_path,
    #     f"gene_tsne_leiden_dim{args.dim_latent}"
    #     f"_lay{args.num_layers}"
    #     f"_epo{args.epochs}"
    #     f"_hid{args.hidden_size}"
    #     f"_perp30_iter2000.png"
    # )

    embedding_tsne = plot_tsne_with_cluster_colors(
        X=gene_saliency,
        cluster_ids=gene_cluster_ids,
        cluster_colors=CLUSTER_COLORS,
        output_path=output_path,
        filename=os.path.basename(tsne_gene_file),
        title="t-SNE of Gene Saliency (Leiden)",
    )

    # embedding = plot_umap_with_cluster_colors(
    #     X,
    #     cluster_ids,
    #     cluster_colors,
    #     output_path,
    #     filename="umap_genes.pdf",
    #     title="Gene Saliency UMAP",
    # )

    plot_cluster_legend(
        CLUSTER_COLORS,
        output_path,
        filename="umap_cluster_legend.png",
        n_rows=3,
    )

    pathway_saliency = saliency_pathway_matrix  # (pathways × genes)
    pathway_cluster_ids = [pathway_clusters[p] for p in pathway_names]

    umap_pathway_file = os.path.join(
        output_path,
        f"pathway_umap_leiden"
        f"_dim{args.dim_latent}"
        f"_lay{args.num_layers}"
        f"_hid{args.hidden_size}"
        f"_epo{args.epochs}.png"
    )
    plot_umap_with_cluster_colors(
        X=pathway_saliency,
        cluster_ids=pathway_cluster_ids,
        cluster_colors=CLUSTER_COLORS,
        output_path=output_path,
        filename=os.path.basename(umap_pathway_file),
        title="Pathway Saliency UMAP",
        point_size=30,
    )

    # embedding_tsne = plot_tsne_with_cluster_colors(
    #     X=pathway_saliency,
    #     cluster_ids=pathway_cluster_ids,
    #     cluster_colors=CLUSTER_COLORS,
    #     output_path=output_path,
    #     filename="pathway_tsne_leiden_tuning.png",
    #     title="t-SNE of Pathway Saliency (Leiden)",
    #     perplexity=30,        # 20–50 typical
    #     n_iter=2000,          # smoother convergence
    #     metric="cosine",      # important for saliency vectors
    #     random_state=42,
    # )

    tsne_pathway_file = os.path.join(
        output_path,
        f"pathway_tsne_leiden"
        f"_dim{args.dim_latent}"
        f"_lay{args.num_layers}"
        f"_hid{args.hidden_size}"
        f"_epo{args.epochs}.png"
    )
    
    embedding_tsne = plot_tsne_with_cluster_colors(
        X=pathway_saliency,
        cluster_ids=pathway_cluster_ids,
        cluster_colors=CLUSTER_COLORS,
        output_path=output_path,
        filename=os.path.basename(tsne_pathway_file),
        title="t-SNE of Pathway Saliency (Leiden)",
    )

    plot_top_genes_per_cluster(
        gene_saliency=saliency_np,
        gene_names=gene_names,
        gene_clusters=gene_clusters,
        output_path=output_path,
        top_k=10
    )

    # plot_gene_umap(
    #     gene_embeddings=h_test.detach().cpu().numpy(),
    #     gene_names=gene_names,
    #     gene_clusters=gene_clusters,
    #     output_path=output_path
    # )

    # gene_cluster_map = build_gene_cluster_map(gene_clusters)

    # pd.DataFrame.from_dict(
    #     gene_clusters,
    #     orient="index",
    #     columns=["LeidenCluster"]
    # ).to_csv(
    #     os.path.join(output_path, "gene_leiden_clusters.csv")
    # )

    # pd.DataFrame.from_dict(
    #     pathway_clusters,
    #     orient="index",
    #     columns=["LeidenCluster"]
    # ).to_csv(
    #     os.path.join(output_path, "pathway_leiden_clusters.csv")
    # )

    # # return {
    # #     "gene_saliency": saliency_np,
    # #     "saliency_pathway_matrix": saliency_pathway_matrix,
    # #     "gene_clusters": gene_clusters,
    # #     "pathway_clusters": pathway_clusters,
    # #     "gene_cluster_map": gene_cluster_map
    # # }



    # edge_ig_norm = (edge_ig - edge_ig.min()) / (edge_ig.max() - edge_ig.min() + 1e-9)

    # edge_list = list(zip(
    #     u_all.cpu().numpy(),
    #     v_all.cpu().numpy(),
    #     edge_ig_norm
    # ))

    # edge_list.sort(key=lambda x: x[2], reverse=True)

    # pd.DataFrame(
    #     edge_list,
    #     columns=["Gene_u", "Gene_v", "IG_Score"]
    # ).to_csv(
    #     os.path.join(output_path, "edge_attributions_all.csv"),
    #     index=False
    # )


    # ########################################
    # # 6. PATHWAY ENRICHMENT MATRIX
    # ########################################

    # df_enrich = pd.read_csv(
    #         "../gene_pathway_embedding/data/processed/gene_gene_pairs_with_pathwayA_enrichment_network.csv"
    # )

    # df_enrich = df_enrich[df_enrich["significance"] == "significant"]
    # df_enrich["enrich_score"] = -np.log10(df_enrich["pvalue"])

    # gene_names = sorted(df_enrich["Gene2"].unique())
    # pathway_names = sorted(df_enrich["PathwayB"].unique())

    # gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    # pathway_to_idx = {p: i for i, p in enumerate(pathway_names)}

    # enrich_matrix = np.zeros((len(gene_names), len(pathway_names)))

    # df_enrich = df_enrich[df_enrich["significance"] == "significant"]
    # df_enrich["enrich_score"] = -np.log10(df_enrich["pvalue"])

    # # enrich_matrix, gene_names, pathway_names = load_or_build_enrichment_matrix(
    # #     df_enrich,
    # #     output_path
    # # )


    # saliency_pathway_matrix = np.zeros(
    #     (len(pathway_names), len(gene_names))
    # )



    # ########################################
    # # 8. BICLUSTERING ON EXPLANATIONS
    # ########################################

    # # pathway_clusters, gene_clusters, leiden_labels = leiden_bipartite_from_saliency(
    # #     saliency_pathway_matrix,
    # #     gene_names,
    # #     pathway_names,
    # #     resolution=1.2,
    # #     weight_threshold=np.percentile(saliency_pathway_matrix, 75)
    # # )

    # def build_gene_cluster_map(gene_clusters):
    #     m = defaultdict(list)
    #     for gene, c in gene_clusters.items():
    #         m[c].append(gene)
    #     return dict(m)


    # # gene_cluster_map = build_gene_cluster_map(gene_clusters)


    # bicluster = SpectralBiclustering(
    #     n_clusters=(10, 10),
    #     method="log",
    #     random_state=0
    # )

    # bicluster.fit(saliency_pathway_matrix)

    # row_labels = bicluster.row_labels_
    # col_labels = bicluster.column_labels_


    
    # save_top_gene_pathway_pairs(saliency_pathway_matrix, gene_names, pathway_names, output_path)
    # save_cluster_pathway_gene_flows(saliency_pathway_matrix, row_labels, gene_names, pathway_names, output_path)
    # gene_clusters = assign_gene_clusters(gene_names, col_labels)
    # gene_cluster_map = extract_gene_cluster_map(saliency_pathway_matrix, row_labels)

    # # -------------------------------
    # # 7. Expression & survival analysis
    # # -------------------------------
    # surv = load_survival("../ACGNN/data/TCGA-BRCA.survival.tsv")
    # expr_matrix, expr_df, surv, patient_ids, common_genes, gene_to_expr_idx = \
    # preprocess_expression(
    #     "../ACGNN/data/TCGA-BRCA.expression.tsv",
    #     surv,
    #     gene_names
    # )


    # # expr_matrix, expr_df, surv, patient_ids, common_genes, gene_to_expr_idx = preprocess_expression(expr_df, surv, gene_names)

    # patient_cluster_scores = compute_patient_cluster_scores(saliency_pathway_matrix, row_labels, gene_names, gene_to_expr_idx, expr_matrix)
    # df_surv = surv.join(patient_cluster_scores, how="inner")

    # # evaluate_survival(df_surv, output_path)
    # plot_patient_cluster_heatmap(patient_cluster_scores, surv, output_path)
    # df_surv = add_high_low_groups(df_surv)

    # # for c in patient_cluster_scores.columns:
    # #     plot_km(df_surv, c, output_path)

    # patient_bicluster = patient_cluster_scores.idxmax(axis=1).loc[df_surv.index]
    # # plot_patient_bicluster_heatmap(patient_cluster_scores, patient_bicluster, output_path)

    # # -------------------------------
    # # 8. Gene–Pathway modules, UMAP, DCA, Risk plots, Nelson-Aalen, Cox
    # # -------------------------------
    # build_and_plot_gene_pathway_modules_centered(df_enrich, os.path.join(output_path, "pathway_centered_gene_modules"), top_genes_per_pathway=50, top_pathways=4)
    # # umap_patients(patient_cluster_scores, df_surv, output_path)
    # # dca_patients(df_surv, patient_cluster_scores, output_path, decision_curve_analysis)
    # # risk_violin_plots(df_surv, patient_cluster_scores, output_path)
    # # nelson_aalen_plots(df_surv, patient_cluster_scores.columns, output_path)
    # # cox_univariate(df_surv, patient_cluster_scores, output_path)

    # # # -------------------------------
    # # # 9. Patient × pathway/family scores & Cox
    # # # -------------------------------
    # # patient_pathway_scores, patient_family_scores, df_family_surv = patient_pathway_family_scores(
    # #     saliency_pathway_matrix, gene_names, gene_to_expr_idx, expr_matrix, patient_ids, pathway_names, output_path
    # # )
    # # cox_pathway_family(patient_family_scores, df_family_surv, output_path)

    # # # -------------------------------
    # # # 10. Gene–pathway heatmaps and modules
    # # # -------------------------------
    # # plot_gene_pathway_modules(df_enrich, output_path)
    # # gene_pathway_heatmaps(saliency_pathway_matrix, gene_names, pathway_names, patient_cluster_scores, row_labels, col_labels, output_path)
    # # km_pathway_family(df_family_surv, patient_family_scores, output_path)

    # # -------------------------------
    # # 11. Sankey plots
    # # -------------------------------
    # df_flows = pd.read_csv(os.path.join(output_path, "sankey_cluster_pathway_gene.csv"))
    # # CLUSTER_COLORS = {"Cluster 0": "#0077B6", "Cluster 1": "#00B4D8", "Cluster 2": "#48CAE4", "Cluster 3": "#90DBF4"}
    # PATHWAY_FAMILY_MAP = {
    #     "PI3K-AKT signaling pathway": "PI3K/AKT",
    #     "mTOR signaling pathway": "PI3K/AKT",
    #     "MAPK signaling pathway": "MAPK",
    #     "ERK cascade": "MAPK",
    #     "p53 signaling pathway": "Cell Cycle / DNA Damage",
    #     "Cell cycle": "Cell Cycle / DNA Damage",
    #     "DNA repair": "Cell Cycle / DNA Damage",
    # }

    # # plot_sankey_all(df_flows, output_path, cluster_colors=CLUSTER_COLORS)
    # # for cluster in sorted(df_flows["Cluster"].unique()):
    # #     df_c = df_flows[df_flows["Cluster"] == cluster]
    # #     plot_cluster_sankey(df_c, cluster, output_path, cluster_colors=CLUSTER_COLORS, pathway_family_map=PATHWAY_FAMILY_MAP)

    # gene_cluster_map = map_genes_to_clusters(saliency_pathway_matrix, bicluster, top_k=20)
    # # plot_km_clusters(df_surv, patient_cluster_scores, output_path)

    # print("✅ train_and_evaluate completed.")

    # # return {
    # #     "embeddings": embeddings,
    # #     "gene_saliency": gene_saliency,
    # #     "saliency_pathway_matrix": saliency_pathway_matrix,
    # #     "gene_cluster_map": gene_cluster_map,
    # #     "gene_clusters": gene_clusters
    # # }










def sort_gene_clusters_by_saliency(
    saliency_matrix, gene_names, gene_clusters
):
    scores = {}
    for g, gname in enumerate(gene_names):
        cid = gene_clusters[gname]
        scores.setdefault(cid, 0.0)
        scores[cid] += saliency_matrix[:, g].sum()

    return sorted(scores, key=scores.get, reverse=True)



def sort_pathway_clusters_by_saliency(
    saliency_matrix, pathway_names, pathway_clusters
):
    scores = {}
    for p, pname in enumerate(pathway_names):
        cid = pathway_clusters[pname]
        scores.setdefault(cid, 0.0)
        scores[cid] += saliency_matrix[p, :].sum()

    return sorted(scores, key=scores.get, reverse=True)

def sort_pathways_within_clusters(
    saliency_matrix, pathway_names, pathway_clusters, sorted_clusters
):
    order = []

    for cid in sorted_clusters:
        members = [
            (p, pname)
            for p, pname in enumerate(pathway_names)
            if pathway_clusters[pname] == cid
        ]

        members_sorted = sorted(
            members,
            key=lambda x: saliency_matrix[x[0], :].sum(),
            reverse=True,
        )

        order.extend([idx for idx, _ in members_sorted])

    return order

def sort_genes_within_clusters(
    saliency_matrix, gene_names, gene_clusters, sorted_clusters
):
    order = []

    for cid in sorted_clusters:
        members = [
            (g, gname)
            for g, gname in enumerate(gene_names)
            if gene_clusters[gname] == cid
        ]

        members_sorted = sorted(
            members,
            key=lambda x: saliency_matrix[:, x[0]].sum(),
            reverse=True,
        )

        order.extend([idx for idx, _ in members_sorted])

    return order


def choose_model(model_type, in_feats, hidden_feats, out_feats):
    if model_type == 'GraphSAGE':
        return GraphSAGE(in_feats, hidden_feats, out_feats)
    elif model_type == 'GAT':
        return GAT(in_feats, hidden_feats, out_feats, num_heads=1)
    elif model_type == 'GCN':
        return GCN(in_feats, hidden_feats, out_feats)
    elif model_type == 'GIN':
        return GIN(in_feats, hidden_feats, out_feats)
    elif model_type == 'HGDC':
        return GAT(in_feats, hidden_feats, out_feats, num_heads=1)
    elif model_type == 'EMOGI':
        return GAT(in_feats, hidden_feats, out_feats, num_heads=1)
    elif model_type == 'MTGCN':
        return GCN(in_feats, hidden_feats, out_feats)
    elif model_type == 'ChebNet':
        return ChebNet(in_feats, hidden_feats, out_feats)
    elif model_type == 'ChebNetII':
        return ChebNet(in_feats, hidden_feats, out_feats)    
    elif model_type == 'DMGNN':
        return DMGNN(
            in_feat_dim=in_feats,
            hidden_dim=hidden_feats,
            out_dim=out_feats,
            heads=4,
            dropout=0.5
        )
    elif model_type == 'ACGNN':
        return ACGNN(in_feats, hidden_feats, out_feats)
    else:
        raise ValueError("Invalid model type. Choose from ['GraphSAGE', 'GAT', 'EMOGI', 'HGDC', 'MTGCN', 'GCN', 'GIN', 'ChebNet', 'ChebNetII', 'DMGNN', 'ACGNN'].")

def build_group_sankey(
    group_name,
    df_surv,
    expr_matrix,
    patient_ids,
    gene_names,
    pathway_names,
    gene_to_expr_idx,
    gene_clusters,
    pathway_clusters,
    saliency_pathway_matrix,
    PATHWAY_FAMILY_MAP,
    top_k=20,
):
    """
    Build GeneCluster → PathwayCluster → Family Sankey edges
    for a survival group (High / Low)
    """

    # -----------------------------
    # select patients
    # -----------------------------
    group_patients = df_surv.index[
        df_surv[f"{group_name}_group"] == group_name
    ]

    patient_mask = [
        patient_ids.index(p)
        for p in group_patients
        if p in patient_ids
    ]

    if len(patient_mask) == 0:
        raise ValueError(f"No patients found for group {group_name}")

    expr_grp = expr_matrix[patient_mask]  # patients × genes

    # -----------------------------
    # pathway → gene map (top-K)
    # -----------------------------
    rows = []

    for pi, pathway in enumerate(pathway_names):
        p_cluster = f"P{pathway_clusters[pathway]}"
        family = PATHWAY_FAMILY_MAP.get(pathway, "Other")

        scores = saliency_pathway_matrix[pi]
        top_genes = np.argsort(scores)[::-1][:top_k]

        for gi in top_genes:
            gene = gene_names[gi]
            if gene not in gene_to_expr_idx:
                continue

            g_cluster = f"G{gene_clusters[gene]}"
            expr_idx = gene_to_expr_idx[gene]

            # expression-weighted saliency
            weight = scores[gi] * expr_grp[:, expr_idx].mean()
            if weight <= 0:
                continue

            rows.append({
                "gene_cluster": g_cluster,
                "pathway_cluster": p_cluster,
                "family": family,
                "weight": weight,
            })

    df = pd.DataFrame(rows)

    # -----------------------------
    # aggregate flows
    # -----------------------------
    df_gp = (
        df.groupby(["gene_cluster", "pathway_cluster"])["weight"]
        .sum()
        .reset_index()
    )

    df_pf = (
        df.groupby(["pathway_cluster", "family"])["weight"]
        .sum()
        .reset_index()
    )

    return df_gp, df_pf

def sankey_node_color(label):
    if label.startswith("G"):
        return "#4C72B0"   # gene clusters
    if label.startswith("P"):
        return "#DD8452"   # pathway clusters
    return "#55A868"      # families

def log_scale_weights(values, base=10):
    """
    Log-scale edge weights for Sankey plots.
    Preserves ordering and avoids zero/negative issues.
    """
    values = np.array(values, dtype=float)
    values[values < 0] = 0.0

    if base == 10:
        return np.log10(values + 1.0)
    elif base == 2:
        return np.log2(values + 1.0)
    else:
        return np.log(values + 1.0)

def plot_umap_with_cluster_colors_legend_on_the_right(
    X,
    cluster_ids,
    cluster_colors,
    output_path,
    filename,
    title,
    n_neighbors=15,
    min_dist=0.2,
    metric="cosine",
    point_size=8,
    alpha=0.85,
    random_state=42,
):
    """
    UMAP embedding with colors aligned to Leiden axis bars.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, n_features), e.g. gene or pathway saliency vectors
    cluster_ids : list or np.ndarray
        Cluster assignment per sample (Leiden)
    cluster_colors : dict
        {cluster_id: hex_color} — SAME dict used for axis bars
    output_path : str
    filename : str
    title : str
    """

    # --- Normalize (important for saliency) ---
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    # --- UMAP ---
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X)

    # --- Exact color alignment ---
    colors = [cluster_colors[c] for c in cluster_ids]

    # --- Plot ---
    plt.figure(figsize=(7, 6))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        s=point_size,
        alpha=alpha,
        linewidths=0,
    )

    # --- Legend (cluster colors, outside plot) ---
    unique_clusters = []
    for c in cluster_ids:
        if c not in unique_clusters:
            unique_clusters.append(c)

    legend_patches = [
        Patch(facecolor=cluster_colors[c], edgecolor="none", label=f"Cluster {c}")
        for c in unique_clusters
    ]

    plt.legend(
        handles=legend_patches,
        title="Cluster",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),  # <-- outside right
        frameon=False,
        fontsize=9,
        title_fontsize=10,
        ncol=1,
    )

    plt.xlabel("UMAP-1", fontsize=14)
    plt.ylabel("UMAP-2", fontsize=14)
    plt.title(title, fontsize=15)

    # Leave room on the right for legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.savefig(
        os.path.join(output_path, filename),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


    return embedding

def plot_umap_with_cluster_colors(
    X,
    cluster_ids,
    cluster_colors,
    output_path,
    filename,
    title,
    n_neighbors=15,
    min_dist=0.2,
    metric="cosine",
    point_size=30,
    alpha=0.85,
    random_state=42,
):

    # --- Normalize ---
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    # --- UMAP ---
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X)

    # --- Colors ---
    colors = [cluster_colors[c] for c in cluster_ids]

    # --- Plot ---
    plt.figure(figsize=(7, 6))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        s=point_size,
        alpha=alpha,
        linewidths=0,
    )

    plt.xlabel("UMAP-1", fontsize=14)
    plt.ylabel("UMAP-2", fontsize=14)
    plt.title(title, fontsize=15)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, filename),
        dpi=300,
    )
    plt.close()

    return embedding



def plot_tsne_with_cluster_colors_anomiely(
    X,
    cluster_ids,
    cluster_colors,
    output_path,
    filename,
    title,
    perplexity=30,
    learning_rate="auto",
    n_iter=1000,
    metric="cosine",
    point_size=8,
    alpha=0.85,
    random_state=42,
):
    """
    t-SNE embedding with colors aligned to Leiden axis bars.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, n_features)
    cluster_ids : list or np.ndarray
        Cluster assignment per sample (Leiden)
    cluster_colors : dict
        {cluster_id: color} — same dict used elsewhere
    """

    # --- Normalize (important for saliency vectors) ---
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    # --- t-SNE ---
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        metric=metric,
        random_state=random_state,
        init="pca",          # stable initialization
        verbose=0,
    )

    embedding = tsne.fit_transform(X)

    # --- Colors ---
    colors = [cluster_colors[c] for c in cluster_ids]

    # --- Plot ---
    plt.figure(figsize=(7, 6))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        s=point_size,
        alpha=alpha,
        linewidths=0,
    )

    plt.xlabel("t-SNE 1", fontsize=14)
    plt.ylabel("t-SNE 2", fontsize=14)
    plt.title(title, fontsize=15)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, filename),
        dpi=300,
    )
    plt.close()

    return embedding

def plot_tsne_with_cluster_colors(
    X,
    cluster_ids,
    cluster_colors,
    output_path,
    filename,
    title,
    perplexity=30,
    learning_rate="auto",
    n_iter=1000,
    metric="cosine",
    point_size=30,
    alpha=0.85,
    random_state=42,
    min_strength_percentile=5,   # ← NEW
):
    """
    t-SNE embedding with Leiden colors and low-saliency filtering.
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # ==========================================================
    # 1. Saliency strength (L2 norm)
    # ==========================================================
    strengths = np.linalg.norm(X, axis=1)

    thr = np.percentile(strengths, min_strength_percentile)

    mask = strengths > thr

    X = X[mask]
    cluster_ids = np.array(cluster_ids)[mask]

    # ==========================================================
    # 2. Normalize (important for cosine metric)
    # ==========================================================
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    # ==========================================================
    # 3. t-SNE
    # ==========================================================
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        metric=metric,
        random_state=random_state,
        init="pca",
        verbose=0,
    )

    embedding = tsne.fit_transform(X)

    # ==========================================================
    # 4. Colors
    # ==========================================================
    colors = [cluster_colors[c] for c in cluster_ids]

    # ==========================================================
    # 5. Plot
    # ==========================================================
    plt.figure(figsize=(7, 6))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        s=point_size,
        alpha=alpha,
        linewidths=0,
    )

    plt.xlabel("t-SNE 1", fontsize=14)
    plt.ylabel("t-SNE 2", fontsize=14)
    plt.title(title, fontsize=15)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, filename),
        dpi=300,
    )
    plt.close()

    return embedding

def plot_cluster_legend(
    cluster_colors,
    output_path,
    filename="cluster_legend.pdf",
    n_rows=3,
    title="Cluster",
    fontsize=10,
    title_fontsize=11,
):
    """
    Create a standalone legend plot with patches arranged in n_rows.
    """

    clusters = list(cluster_colors.keys())
    n_clusters = len(clusters)
    n_cols = math.ceil(n_clusters / n_rows)

    legend_patches = [
        Patch(facecolor=cluster_colors[c], edgecolor="none", label=f"Cluster {c}")
        for c in clusters
    ]

    fig, ax = plt.subplots(figsize=(1.6 * n_cols, 0.6 * n_rows))
    ax.axis("off")

    ax.legend(
        handles=legend_patches,
        title=title,
        ncol=n_cols,
        loc="center",
        frameon=False,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
        handlelength=1.2,
        handleheight=1.2,
        columnspacing=1.4,
        labelspacing=0.8,
    )

    os.makedirs(output_path, exist_ok=True)
    plt.savefig(
        os.path.join(output_path, filename),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def load_or_compute_relevance(
    model,
    graph,
    features,
    method="saliency",
    node_indices=None,
    use_abs=True,
    baseline=None,
    steps=50,
    cache_dir="results/relevance",
    force_recompute=False
):
    os.makedirs(cache_dir, exist_ok=True)

    fname = f"relevance_{method}.pt"
    fpath = os.path.join(cache_dir, fname)

    meta_path = os.path.join(cache_dir, "meta.json")

    # ---------- LOAD ----------
    if os.path.exists(fpath) and not force_recompute:
        print(f"📦 Loading cached relevance: {fpath}")
        relevance = torch.load(fpath, map_location="cpu")

        # Optional sanity check
        if relevance.shape[1] != features.shape[1]:
            raise ValueError("Cached relevance feature dim mismatch")

        return relevance

    # ---------- COMPUTE ----------
    print(f"🧠 Computing relevance ({method}) ...")
    relevance = compute_relevance_scores(
        model=model,
        graph=graph,
        features=features,
        node_indices=node_indices,
        method=method,
        use_abs=use_abs,
        baseline=baseline,
        steps=steps
    )

    # ---------- SAVE ----------
    torch.save(relevance.cpu(), fpath)

    meta = {
        "method": method,
        "use_abs": use_abs,
        "steps": steps,
        "num_nodes": relevance.shape[0],
        "num_features": relevance.shape[1]
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"💾 Relevance saved to {fpath}")

    return relevance

def load_or_build_enrichment_matrix(df_enrich, output_path):
    os.makedirs(output_path, exist_ok=True)

    mat_path = os.path.join(output_path, "enrich_matrix.npy")
    gene_path = os.path.join(output_path, "enrich_genes.npy")
    pathway_path = os.path.join(output_path, "enrich_pathways.npy")

    if (
        os.path.exists(mat_path) and
        os.path.exists(gene_path) and
        os.path.exists(pathway_path)
    ):
        print("✅ Loading saved enrichment matrix")
        enrich_matrix = np.load(mat_path)
        gene_names = np.load(gene_path, allow_pickle=True).tolist()
        pathway_names = np.load(pathway_path, allow_pickle=True).tolist()
        return enrich_matrix, gene_names, pathway_names

    print("🔄 Building enrichment matrix")

    gene_names = sorted(df_enrich["Gene2"].astype(str).unique())
    pathway_names = sorted(df_enrich["PathwayB"].astype(str).unique())

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    pathway_to_idx = {p: i for i, p in enumerate(pathway_names)}

    enrich_matrix = np.zeros(
        (len(gene_names), len(pathway_names)),
        dtype=np.float32
    )

    for _, row in tqdm(
        df_enrich.iterrows(),
        total=len(df_enrich),
        desc="Building enrichment matrix"
    ):
        enrich_matrix[
            gene_to_idx[str(row["Gene2"])],
            pathway_to_idx[str(row["PathwayB"])]
        ] = row["enrich_score"]

    np.save(mat_path, enrich_matrix)
    np.save(gene_path, np.array(gene_names, dtype=object))
    np.save(pathway_path, np.array(pathway_names, dtype=object))

    print("✅ Enrichment matrix saved")

    return enrich_matrix, gene_names, pathway_names

def edge_integrated_gradients(
    h,
    u_idx,
    v_idx,
    predictor,
    steps=50
):
    """
    h          : torch.Tensor (N × d) node embeddings
    u_idx,v_idx: edge endpoint indices (E,)
    predictor  : trained MLPPredictor
    steps      : IG steps
    """

    device = h.device

    edge_embed = torch.cat([h[u_idx], h[v_idx]], dim=1)
    baseline = torch.zeros_like(edge_embed)

    total_grad = torch.zeros_like(edge_embed)

    alphas = torch.linspace(0, 1, steps, device=device)

    for alpha in tqdm(
        alphas,
        desc="Edge Integrated Gradients",
        leave=True
    ):
        interp = baseline + alpha * (edge_embed - baseline)
        interp.requires_grad_(True)

        predictor.zero_grad(set_to_none=True)

        score = predictor.forward_from_embedding(interp).sum()
        score.backward()

        total_grad += interp.grad.detach()

    avg_grad = total_grad / steps
    ig = (edge_embed - baseline) * avg_grad

    return ig.abs().sum(dim=1)

def leiden_pathway_only(saliency_pathway_matrix, pathway_names, resolution=1.0):
    sim = np.corrcoef(saliency_pathway_matrix)
    edges, weights = [], []

    for i in range(len(pathway_names)):
        for j in range(i + 1, len(pathway_names)):
            if sim[i, j] > 0.3:
                edges.append((i, j))
                weights.append(sim[i, j])

    g = ig.Graph(n=len(pathway_names), edges=edges)
    g.es["weight"] = weights

    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution
    )

    return {
        pathway_names[i]: part.membership[i]
        for i in range(len(pathway_names))
    }

def leiden_bipartite_from_saliency_(
    saliency_pathway_matrix,
    gene_names,
    pathway_names,
    resolution=1.0,
    weight_threshold=0.0
):
    n_pathways, n_genes = saliency_pathway_matrix.shape

    edges = []
    weights = []

    for p in range(n_pathways):
        for g in range(n_genes):
            w = saliency_pathway_matrix[p, g]
            if w > weight_threshold:
                edges.append((p, n_pathways + g))
                weights.append(float(w))

    g = ig.Graph(
        n=n_pathways + n_genes,
        edges=edges,
        edge_attrs={"weight": weights}
    )

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution
    )

    labels = np.array(partition.membership)

    pathway_clusters = {
        pathway_names[i]: labels[i]
        for i in range(n_pathways)
    }

    gene_clusters = {
        gene_names[j]: labels[n_pathways + j]
        for j in range(n_genes)
    }

    return pathway_clusters, gene_clusters, labels

def plot_leiden_saliency_heatmap_ori(
    saliency_pathway_matrix,
    gene_names,
    pathway_names,
    gene_clusters,
    pathway_clusters,
    output_path,
    vmax_percentile=99
):
    gene_order = sorted(
        range(len(gene_names)),
        key=lambda i: gene_clusters[gene_names[i]]
    )

    pathway_order = sorted(
        range(len(pathway_names)),
        key=lambda i: pathway_clusters[pathway_names[i]]
    )

    mat = saliency_pathway_matrix[np.ix_(pathway_order, gene_order)]

    vmax = np.percentile(mat, vmax_percentile)

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        mat,
        cmap="mako",
        vmax=vmax,
        xticklabels=False,
        yticklabels=False
    )

    plt.xlabel("Genes (Leiden ordered)")
    plt.ylabel("Pathways (Leiden ordered)")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "leiden_gene_pathway_heatmap.png"),
        dpi=300
    )
    plt.close()


def plot_leiden_saliency_heatmap_axisbar(
    saliency_pathway_matrix,
    gene_names,
    pathway_names,
    gene_clusters,
    pathway_clusters,
    output_path,
    vmax_percentile=99,
    sort_by_cluster=True,
    gamma=0.35,
    cmap="inferno",
    filename="leiden_gene_pathway_heatmap_axisbars.png",
    figsize=(18, 12),
):
    """
    Gene–pathway saliency heatmap with Leiden ordering, axis cluster bars,
    and log-interpretable PowerNorm colorbar.
    """


    # ==========================================================
    # 1. Ordering
    # ==========================================================
    if sort_by_cluster:
        gene_order = sorted(
            range(len(gene_names)),
            key=lambda i: gene_clusters[gene_names[i]]
        )
        pathway_order = sorted(
            range(len(pathway_names)),
            key=lambda i: pathway_clusters[pathway_names[i]]
        )
    else:
        gene_order = list(range(len(gene_names)))
        pathway_order = list(range(len(pathway_names)))

    mat = saliency_pathway_matrix[np.ix_(pathway_order, gene_order)]

    # ==========================================================
    # 2. PowerNorm normalization (safe for zeros)
    # ==========================================================
    nonzero = mat[mat > 0]
    vmin = np.percentile(nonzero, 3) if nonzero.size else 1e-6
    vmin = max(vmin, 1e-6)
    vmax = np.percentile(mat, vmax_percentile)

    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    
    # ==========================================================
    # 3. Cluster color bars (fixed palette)
    # ==========================================================
    gene_cluster_ids = [gene_clusters[gene_names[i]] for i in gene_order]
    pathway_cluster_ids = [pathway_clusters[pathway_names[i]] for i in pathway_order]

    def cluster_to_rgb(cid):
        if cid not in CLUSTER_COLORS:
            raise ValueError(f"Cluster ID {cid} not in CLUSTER_COLORS")
        return to_rgb(CLUSTER_COLORS[cid])

    gene_colors = np.array([cluster_to_rgb(c) for c in gene_cluster_ids])
    pathway_colors = np.array([cluster_to_rgb(c) for c in pathway_cluster_ids])


    # ==========================================================
    # 4. Layout (explicit colorbar axis)
    # ==========================================================
    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        2, 3,
        width_ratios=[0.02, 0.93, 0.03],   # ← colorbar column
        height_ratios=[0.02, 0.98],
        wspace=0.08,
        hspace=0.02,
    )

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.995)

    ax_left = fig.add_subplot(gs[1, 0])   # pathway bar
    ax_top  = fig.add_subplot(gs[0, 1])   # gene bar
    ax_main = fig.add_subplot(gs[1, 1])   # heatmap
    ax_cbar = fig.add_subplot(gs[1, 2])   # colorbar

    # ==========================================================
    # 5. Heatmap
    # ==========================================================
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_under("#1f1f1f")
    cmap_obj.set_bad("#1f1f1f")

    mat_plot = np.ma.masked_where(mat <= 0, mat)
    # ---- soften contrast (gamma correction) ----
    GAMMA = 0.4   # 0.3–0.6 works well

    mat_plot = np.power(mat / vmax, GAMMA)

    hm = sns.heatmap(
        mat_plot,
        ax=ax_main,
        cmap=cmap_obj,
        norm=norm,
        xticklabels=False,
        yticklabels=False,
        linewidths=0,
        rasterized=True,
        cbar=False,
    )

    ax_main.set_facecolor("#1f1f1f")

    # ==========================================================
    # 6. Colorbar (log-interpretable)
    # ==========================================================
    cbar = fig.colorbar(
        hm.collections[0],
        cax=ax_cbar,
    )

    # shorten & center
    pos = ax_cbar.get_position()
    new_h = pos.height * 0.7
    ax_cbar.set_position([
        pos.x0,
        pos.y0 + (pos.height - new_h) / 2,
        pos.width,
        new_h,
    ])

    cbar.set_label("Saliency", fontsize=18, labelpad=16)
    cbar.ax.tick_params(labelsize=14, width=0.8)

    # ---- log-like ticks ----
    log_min = np.log10(norm.vmin)
    log_max = np.log10(norm.vmax)

    log_ticks = np.linspace(
        np.floor(log_min),
        np.ceil(log_max),
        num=min(5, int(np.ceil(log_max - log_min)) + 1),
    )

    tick_vals = 10 ** log_ticks
    tick_vals = tick_vals[
        (tick_vals >= norm.vmin) & (tick_vals <= norm.vmax)
    ]

    cbar.set_ticks(tick_vals)

    def log_fmt(val, pos=None):
        return rf"$10^{{{int(np.round(np.log10(val)))}}}$"

    cbar.ax.yaxis.set_major_formatter(FuncFormatter(log_fmt))
    cbar.outline.set_linewidth(0.8)

    # ==========================================================
    # 7. Axis color bars
    # ==========================================================
    ax_top.imshow(gene_colors[None, :, :], aspect="auto")
    ax_top.set_axis_off()

    ax_left.imshow(pathway_colors[:, None, :], aspect="auto")
    ax_left.set_axis_off()

    # force alignment
    fig.canvas.draw()
    main_pos = ax_main.get_position()
    top_pos  = ax_top.get_position()

    ax_top.set_position([
        main_pos.x0,
        top_pos.y0,
        main_pos.width,
        top_pos.height,
    ])

    # ==========================================================
    # 8. Save
    # ==========================================================
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(
        os.path.join(output_path, filename),
        dpi=300,
    )
    plt.close()


def plot_leiden_saliency_heatmap_black(
    saliency_pathway_matrix,
    gene_names,
    pathway_names,
    gene_clusters,
    pathway_clusters,
    output_path,
    vmax_percentile=99
):
    gene_order = sorted(
        range(len(gene_names)),
        key=lambda i: gene_clusters[gene_names[i]]
    )

    pathway_order = sorted(
        range(len(pathway_names)),
        key=lambda i: pathway_clusters[pathway_names[i]]
    )

    mat = saliency_pathway_matrix[np.ix_(pathway_order, gene_order)]

    vmax = np.percentile(mat, vmax_percentile)

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        mat,
        cmap="mako",
        vmax=vmax,
        xticklabels=False,
        yticklabels=False
    )

    plt.xlabel("Genes (Leiden ordered)")
    plt.ylabel("Pathways (Leiden ordered)")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "leiden_gene_pathway_heatmap.png"),
        dpi=300
    )
    plt.close()


def plot_leiden_saliency_heatmap_dark_pass(
    saliency_pathway_matrix,
    gene_names,
    pathway_names,
    gene_clusters,
    pathway_clusters,
    output_path,
    vmax_percentile=99,
    figsize=(14, 8),
    filename="leiden_gene_pathway_heatmap_axisbars.png",
):
    import os
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import to_rgb
    import colorcet as cc

    # ==========================================================
    # 1. Leiden ordering
    # ==========================================================
    gene_order = sorted(
        range(len(gene_names)),
        key=lambda i: gene_clusters[gene_names[i]]
    )
    pathway_order = sorted(
        range(len(pathway_names)),
        key=lambda i: pathway_clusters[pathway_names[i]]
    )

    mat = saliency_pathway_matrix[np.ix_(pathway_order, gene_order)]
    vmax = np.percentile(mat, vmax_percentile)

    # ==========================================================
    # 2. Cluster → color mapping
    # ==========================================================
    gene_cluster_ids = [gene_clusters[gene_names[i]] for i in gene_order]
    pathway_cluster_ids = [pathway_clusters[pathway_names[i]] for i in pathway_order]

    def make_cluster_cmap(cluster_ids):
        uniq = []
        for c in cluster_ids:
            if c not in uniq:
                uniq.append(c)
        palette = cc.glasbey[:len(uniq)]
        return dict(zip(uniq, palette))

    gene_cmap = make_cluster_cmap(gene_cluster_ids)
    pathway_cmap = make_cluster_cmap(pathway_cluster_ids)

    gene_colors = np.array([to_rgb(gene_cmap[c]) for c in gene_cluster_ids])
    pathway_colors = np.array([to_rgb(pathway_cmap[c]) for c in pathway_cluster_ids])

    # ==========================================================
    # 3. Layout
    # ==========================================================
    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        2, 3,
        width_ratios=[0.022, 0.92, 0.058],   # extra space for colorbar
        height_ratios=[0.03, 0.97],
        wspace=0.0,
        hspace=0.0,
    )

    ax_top  = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    # ==========================================================
    # 4. Heatmap
    # ==========================================================
    hm = sns.heatmap(
        mat,
        ax=ax_main,
        cmap="mako",
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar=True,
        cbar_ax=ax_cbar,
        rasterized=True,
    )

    ax_main.set_xlabel("")
    ax_main.set_ylabel("")

    cbar = hm.collections[0].colorbar
    cbar.set_label("")

    # ==========================================================
    # 5. Axis cluster bars
    # ==========================================================
    ax_top.imshow(gene_colors[None, :, :], aspect="auto")
    ax_top.set_axis_off()

    ax_left.imshow(pathway_colors[:, None, :], aspect="auto")
    ax_left.set_axis_off()

    # ==========================================================
    # 6. External text labels (clean, non-TeX-bold)
    # ==========================================================
    ax_top.text(
        0.5, 1.9,
        "Genes",
        ha="center",
        va="bottom",
        transform=ax_top.transAxes,
        fontsize=22,
    )

    ax_left.text(
        -1.8, 0.5,
        "Pathways",
        ha="center",
        va="center",
        rotation=90,
        transform=ax_left.transAxes,
        fontsize=22,
    )

    # ==========================================================
    # 7. Alignment + resized colorbar
    # ==========================================================
    fig.canvas.draw()
    main_pos = ax_main.get_position()
    cbar_pos = ax_cbar.get_position()

    # y-axis bar flush
    ax_left.set_position([
        ax_left.get_position().x0,
        main_pos.y0,
        ax_left.get_position().width,
        main_pos.height,
    ])

    # top bar aligned
    ax_top.set_position([
        main_pos.x0,
        ax_top.get_position().y0,
        main_pos.width,
        ax_top.get_position().height,
    ])

    # colorbar: farther, half width & half height
    new_h = main_pos.height * 0.5
    new_w = cbar_pos.width * 0.5

    ax_cbar.set_position([
        cbar_pos.x0 + 0.02,  # ← extra separation from heatmap
        main_pos.y0 + (main_pos.height - new_h) / 2,
        new_w,
        new_h,
    ])

    # ==========================================================
    # 8. Save
    # ==========================================================
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(
        os.path.join(output_path, filename),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()

def plot_leiden_saliency_heatmap_denoised(
    saliency_pathway_matrix,
    gene_names,
    pathway_names,
    gene_clusters,
    pathway_clusters,
    output_path,
    vmax_percentile=99,
    noise_percentile=5,
    gamma=0.35,
    figsize=(14, 8),
    filename="leiden_gene_pathway_heatmap_denoised.png",
):
    """
    Leiden-ordered gene–pathway saliency heatmap with:
      • percentile-based denoising
      • PowerNorm contrast
      • axis cluster bars
      • half-size, spaced colorbar
    """

    import os
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import PowerNorm, to_rgb
    import colorcet as cc

    # ==========================================================
    # 1. Leiden ordering
    # ==========================================================
    gene_order = sorted(
        range(len(gene_names)),
        key=lambda i: gene_clusters[gene_names[i]]
    )
    pathway_order = sorted(
        range(len(pathway_names)),
        key=lambda i: pathway_clusters[pathway_names[i]]
    )

    mat = saliency_pathway_matrix[np.ix_(pathway_order, gene_order)]

    # ==========================================================
    # 2. Denoising (mask weak saliency)
    # ==========================================================
    nonzero = mat[mat > 0]
    if nonzero.size == 0:
        raise ValueError("Saliency matrix contains no positive values.")

    vmin = np.percentile(nonzero, noise_percentile)
    vmax = np.percentile(mat, vmax_percentile)

    vmin = max(vmin, 1e-6)

    mat_plot = np.ma.masked_less(mat, vmin)

    # ==========================================================
    # 3. Normalization + colormap
    # ==========================================================
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    cmap = plt.get_cmap("mako").copy()
    cmap.set_under("#f2f2f2")  # light background
    cmap.set_bad("#f2f2f2")

    # ==========================================================
    # 4. Cluster → color mapping
    # ==========================================================
    gene_cluster_ids = [gene_clusters[gene_names[i]] for i in gene_order]
    pathway_cluster_ids = [pathway_clusters[pathway_names[i]] for i in pathway_order]

    def make_cluster_cmap(cluster_ids):
        uniq = []
        for c in cluster_ids:
            if c not in uniq:
                uniq.append(c)
        palette = cc.glasbey[:len(uniq)]
        return dict(zip(uniq, palette))

    gene_cmap = make_cluster_cmap(gene_cluster_ids)
    pathway_cmap = make_cluster_cmap(pathway_cluster_ids)

    gene_colors = np.array([to_rgb(gene_cmap[c]) for c in gene_cluster_ids])
    pathway_colors = np.array([to_rgb(pathway_cmap[c]) for c in pathway_cluster_ids])

    # ==========================================================
    # 5. Layout
    # ==========================================================
    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        2, 3,
        width_ratios=[0.022, 0.92, 0.058],
        height_ratios=[0.03, 0.97],
        wspace=0.0,
        hspace=0.0,
    )

    ax_top  = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    # ==========================================================
    # 6. Heatmap
    # ==========================================================
    hm = sns.heatmap(
        mat_plot,
        ax=ax_main,
        cmap=cmap,
        norm=norm,
        xticklabels=False,
        yticklabels=False,
        cbar=True,
        cbar_ax=ax_cbar,
        rasterized=True,
    )

    ax_main.set_xlabel("")
    ax_main.set_ylabel("")

    # ==========================================================
    # 7. Axis cluster bars
    # ==========================================================
    ax_top.imshow(gene_colors[None, :, :], aspect="auto")
    ax_top.set_axis_off()

    ax_left.imshow(pathway_colors[:, None, :], aspect="auto")
    ax_left.set_axis_off()

    # ==========================================================
    # 8. External labels
    # ==========================================================
    ax_top.text(
        0.5, 1.9,
        "Genes",
        ha="center",
        va="bottom",
        transform=ax_top.transAxes,
        fontsize=16,
    )

    ax_left.text(
        -1.8, 0.5,
        "Pathways",
        ha="center",
        va="center",
        rotation=90,
        transform=ax_left.transAxes,
        fontsize=16,
    )

    # ==========================================================
    # 9. Align axes + resize colorbar
    # ==========================================================
    fig.canvas.draw()

    main_pos = ax_main.get_position()
    cbar_pos = ax_cbar.get_position()

    # y-bar flush
    ax_left.set_position([
        ax_left.get_position().x0,
        main_pos.y0,
        ax_left.get_position().width,
        main_pos.height,
    ])

    # top bar aligned
    ax_top.set_position([
        main_pos.x0,
        ax_top.get_position().y0,
        main_pos.width,
        ax_top.get_position().height,
    ])

    # colorbar: half width, half height, spaced
    new_h = main_pos.height * 0.5
    new_w = cbar_pos.width * 0.5

    ax_cbar.set_position([
        cbar_pos.x0 + 0.02,
        main_pos.y0 + (main_pos.height - new_h) / 2,
        new_w,
        new_h,
    ])

    # ==========================================================
    # 10. Save
    # ==========================================================
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(
        os.path.join(output_path, filename),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()

def plot_leiden_saliency_heatmap(
    saliency_pathway_matrix,
    gene_names,
    pathway_names,
    gene_clusters,
    pathway_clusters,
    output_path,
    vmax_percentile=99,
    noise_percentile=5,
    gamma=0.35,
    figsize=(14, 8),
    filename="leiden_gene_pathway_heatmap_denoised.png",
):
    """
    Leiden-ordered gene–pathway saliency heatmap with:
      • percentile-based denoising
      • PowerNorm contrast
      • axis cluster bars
      • half-size, spaced colorbar
    """

    import os
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import PowerNorm, to_rgb
    import colorcet as cc

    # ==========================================================
    # 1. Leiden ordering
    # ==========================================================
    gene_order = sorted(
        range(len(gene_names)),
        key=lambda i: gene_clusters[gene_names[i]]
    )
    pathway_order = sorted(
        range(len(pathway_names)),
        key=lambda i: pathway_clusters[pathway_names[i]]
    )

    mat = saliency_pathway_matrix[np.ix_(pathway_order, gene_order)]

    # ==========================================================
    # 2. Denoising (mask weak saliency)
    # ==========================================================
    nonzero = mat[mat > 0]
    if nonzero.size == 0:
        raise ValueError("Saliency matrix contains no positive values.")

    vmin = np.percentile(nonzero, noise_percentile)
    vmax = np.percentile(mat, vmax_percentile)

    vmin = max(vmin, 1e-6)

    mat_plot = np.ma.masked_less(mat, vmin)

    # ==========================================================
    # 3. Normalization + colormap
    # ==========================================================
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    cmap = plt.get_cmap("mako").copy()
    cmap.set_under("#f2f2f2")  # light background
    cmap.set_bad("#f2f2f2")

    # ==========================================================
    # 4. Cluster → color mapping
    # ==========================================================
    gene_cluster_ids = [gene_clusters[gene_names[i]] for i in gene_order]
    pathway_cluster_ids = [pathway_clusters[pathway_names[i]] for i in pathway_order]

    def make_cluster_cmap(cluster_ids):
        uniq = []
        for c in cluster_ids:
            if c not in uniq:
                uniq.append(c)
        palette = cc.glasbey[:len(uniq)]
        return dict(zip(uniq, palette))

    gene_cmap = make_cluster_cmap(gene_cluster_ids)
    pathway_cmap = make_cluster_cmap(pathway_cluster_ids)

    gene_colors = np.array([to_rgb(gene_cmap[c]) for c in gene_cluster_ids])
    pathway_colors = np.array([to_rgb(pathway_cmap[c]) for c in pathway_cluster_ids])

    # ==========================================================
    # 5. Layout
    # ==========================================================
    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        2, 3,
        width_ratios=[0.022, 0.92, 0.058],
        height_ratios=[0.03, 0.97],
        wspace=0.0,
        hspace=0.0,
    )

    ax_top  = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    # ==========================================================
    # 6. Heatmap
    # ==========================================================
    hm = sns.heatmap(
        mat_plot,
        ax=ax_main,
        cmap=cmap,
        norm=norm,
        xticklabels=False,
        yticklabels=False,
        cbar=True,
        cbar_ax=ax_cbar,
        rasterized=True,
    )

    ax_main.set_xlabel("")
    ax_main.set_ylabel("")

    # ==========================================================
    # 7. Axis cluster bars
    # ==========================================================
    ax_top.imshow(gene_colors[None, :, :], aspect="auto")
    ax_top.set_axis_off()

    ax_left.imshow(pathway_colors[:, None, :], aspect="auto")
    ax_left.set_axis_off()

    # ==========================================================
    # 8. External labels
    # ==========================================================
    ax_top.text(
        0.5, 1.9,
        "Genes",
        ha="center",
        va="bottom",
        transform=ax_top.transAxes,
        fontsize=16,
    )

    ax_left.text(
        -1.8, 0.5,
        "Pathways",
        ha="center",
        va="center",
        rotation=90,
        transform=ax_left.transAxes,
        fontsize=16,
    )

    # ==========================================================
    # 9. Align axes + resize colorbar
    # ==========================================================
    fig.canvas.draw()

    main_pos = ax_main.get_position()
    cbar_pos = ax_cbar.get_position()

    # y-bar flush
    ax_left.set_position([
        ax_left.get_position().x0,
        main_pos.y0,
        ax_left.get_position().width,
        main_pos.height,
    ])

    # top bar aligned
    ax_top.set_position([
        main_pos.x0,
        ax_top.get_position().y0,
        main_pos.width,
        ax_top.get_position().height,
    ])

    # colorbar: half width, half height, spaced
    new_h = main_pos.height * 0.5
    new_w = cbar_pos.width * 0.5

    ax_cbar.set_position([
        cbar_pos.x0 + 0.02,
        main_pos.y0 + (main_pos.height - new_h) / 2,
        new_w,
        new_h,
    ])

    # ==========================================================
    # 10. Save
    # ==========================================================
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(
        os.path.join(output_path, filename),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def plot_leiden_saliency_heatmap_dark_red_backgound(
    saliency_pathway_matrix,
    gene_names,
    pathway_names,
    gene_clusters,
    pathway_clusters,
    output_path,
    vmax_percentile=99,
    figsize=(16, 12),
    filename="leiden_gene_pathway_heatmap_axisbars.png",
):

    # ==========================================================
    # 1. Leiden ordering
    # ==========================================================
    gene_order = sorted(
        range(len(gene_names)),
        key=lambda i: gene_clusters[gene_names[i]]
    )
    pathway_order = sorted(
        range(len(pathway_names)),
        key=lambda i: pathway_clusters[pathway_names[i]]
    )

    mat = saliency_pathway_matrix[np.ix_(pathway_order, gene_order)]
    vmax = np.percentile(mat, vmax_percentile)

    # ==========================================================
    # 1.5 Contrast softening (gamma normalization)
    # ==========================================================
    gamma = 0.45   # 0.35–0.6 works well
    mat_plot = np.power(mat / vmax, gamma)

    # ==========================================================
    # 2. Cluster → color mapping
    # ==========================================================
    gene_cluster_ids = [gene_clusters[gene_names[i]] for i in gene_order]
    pathway_cluster_ids = [pathway_clusters[pathway_names[i]] for i in pathway_order]

    def make_cluster_cmap(cluster_ids):
        uniq = []
        for c in cluster_ids:
            if c not in uniq:
                uniq.append(c)
        palette = cc.glasbey[:len(uniq)]
        return dict(zip(uniq, palette))

    gene_cmap = make_cluster_cmap(gene_cluster_ids)
    pathway_cmap = make_cluster_cmap(pathway_cluster_ids)

    gene_colors = np.array([to_rgb(gene_cmap[c]) for c in gene_cluster_ids])
    pathway_colors = np.array([to_rgb(pathway_cmap[c]) for c in pathway_cluster_ids])

    # ==========================================================
    # 3. Layout
    # ==========================================================
    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        2, 3,
        width_ratios=[0.022, 0.92, 0.058],   # extra space for colorbar
        height_ratios=[0.03, 0.97],
        wspace=0.0,
        hspace=0.0,
    )

    ax_top  = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    # ==========================================================
    # 4. Heatmap
    # ==========================================================
    # hm = sns.heatmap(
    #     mat,
    #     ax=ax_main,
    #     cmap="mako",
    #     vmax=vmax,
    #     xticklabels=False,
    #     yticklabels=False,
    #     cbar=True,
    #     cbar_ax=ax_cbar,
    #     rasterized=True,
    # )
    hm = sns.heatmap(
        mat_plot,
        ax=ax_main,
        cmap="rocket",     # lighter than mako
        vmin=0,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
        cbar=True,
        cbar_ax=ax_cbar,
        rasterized=True,
    )


    ax_main.set_xlabel("")
    ax_main.set_ylabel("")

    cbar = hm.collections[0].colorbar
    cbar.set_label("")

    # ==========================================================
    # 5. Axis cluster bars
    # ==========================================================
    ax_top.imshow(gene_colors[None, :, :], aspect="auto")
    ax_top.set_axis_off()

    ax_left.imshow(pathway_colors[:, None, :], aspect="auto")
    ax_left.set_axis_off()

    # ==========================================================
    # 6. External text labels (clean, non-TeX-bold)
    # ==========================================================
    ax_top.text(
        0.5, 1.9,
        "Genes",
        ha="center",
        va="bottom",
        transform=ax_top.transAxes,
        fontsize=22,
    )

    ax_left.text(
        -1.8, 0.5,
        "Pathways",
        ha="center",
        va="center",
        rotation=90,
        transform=ax_left.transAxes,
        fontsize=22,
    )

    # ==========================================================
    # 7. Alignment + resized colorbar
    # ==========================================================
    fig.canvas.draw()
    main_pos = ax_main.get_position()
    cbar_pos = ax_cbar.get_position()

    # y-axis bar flush
    ax_left.set_position([
        ax_left.get_position().x0,
        main_pos.y0,
        ax_left.get_position().width,
        main_pos.height,
    ])

    # top bar aligned
    ax_top.set_position([
        main_pos.x0,
        ax_top.get_position().y0,
        main_pos.width,
        ax_top.get_position().height,
    ])

    # colorbar: farther, half width & half height
    new_h = main_pos.height * 0.5
    new_w = cbar_pos.width * 0.5

    ax_cbar.set_position([
        cbar_pos.x0 + 0.02,  # ← extra separation from heatmap
        main_pos.y0 + (main_pos.height - new_h) / 2,
        new_w,
        new_h,
    ])

    # ==========================================================
    # 8. Save
    # ==========================================================
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(
        os.path.join(output_path, filename),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def plot_leiden_saliency_heatmap_x(
    saliency_pathway_matrix,
    gene_names,
    pathway_names,
    gene_clusters,
    pathway_clusters,
    output_path,
    vmax_percentile=99,
    figsize=(16, 12),
    filename="leiden_gene_pathway_heatmap_axisbars.png",
):

    # ==========================================================
    # 1. Leiden ordering
    # ==========================================================
    gene_order = sorted(
        range(len(gene_names)),
        key=lambda i: gene_clusters[gene_names[i]]
    )
    pathway_order = sorted(
        range(len(pathway_names)),
        key=lambda i: pathway_clusters[pathway_names[i]]
    )

    mat = saliency_pathway_matrix[np.ix_(pathway_order, gene_order)]
    vmax = np.percentile(mat, vmax_percentile)


    # ==========================================================
    # Log-scaled normalization (safe for zeros)
    # ==========================================================
    mat_plot = mat.copy()

    # Mask non-positive values (LogNorm cannot handle ≤ 0)
    mat_plot[mat_plot <= 0] = np.nan

    # Robust limits
    vmin = np.nanpercentile(mat_plot, 3)
    vmax = np.nanpercentile(mat_plot, vmax_percentile)

    # Safety guards
    vmin = max(vmin, 1e-6)
    vmax = max(vmax, vmin * 10)

    norm = LogNorm(vmin=vmin, vmax=vmax)

    # ==========================================================
    # 2. Cluster color bars (fixed palette)
    # ==========================================================
    gene_cluster_ids = [gene_clusters[gene_names[i]] for i in gene_order]
    pathway_cluster_ids = [pathway_clusters[pathway_names[i]] for i in pathway_order]

    def cluster_to_rgb(cid):
        if cid not in CLUSTER_COLORS:
            raise ValueError(f"Cluster ID {cid} not in CLUSTER_COLORS")
        return to_rgb(CLUSTER_COLORS[cid])

    gene_colors = np.array([cluster_to_rgb(c) for c in gene_cluster_ids])
    pathway_colors = np.array([cluster_to_rgb(c) for c in pathway_cluster_ids])


    # ==========================================================
    # 3. Layout
    # ==========================================================
    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        2, 3,
        width_ratios=[0.022, 0.92, 0.058],   # extra space for colorbar
        height_ratios=[0.03, 0.97],
        wspace=0.0,
        hspace=0.0,
    )

    ax_top  = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])



    # ==========================================================
    # Contrast-aware, noise-robust normalization (CANONICAL)
    # ==========================================================

    # ---- 1. Work on a copy (safety) ----
    mat = np.asarray(mat, dtype=float)

    # ---- 2. Extract nonzero signal ----
    nz = mat[mat > 0]

    if nz.size == 0:
        raise ValueError("Saliency matrix contains no positive values.")

    # ---- 3. Remove low-magnitude noise (sparsity control) ----
    NOISE_Q = 65  # recommended range: 60–75
    noise_floor = np.percentile(nz, NOISE_Q)

    mat_clean = mat.copy()
    mat_clean[mat_clean < noise_floor] = 0.0

    # ---- 4. Mask zeros (important for background clarity) ----
    mat_plot = np.ma.masked_where(mat_clean == 0, mat_clean)

    # ---- 5. Robust normalization bounds (post-threshold!) ----
    signal = mat_clean[mat_clean > 0]

    vmin = np.percentile(signal, 5)      # darker background
    vmax = np.percentile(signal, 99.5)   # suppress extreme outliers

    # Safety fallback
    vmin = max(vmin, 1e-6)

    # ---- 6. Power-law normalization (contrast boost) ----
    norm = PowerNorm(
        gamma=0.45,   # 0.4–0.6 sweet spot for saliency
        vmin=vmin,
        vmax=vmax,
        clip=True,
    )

    # ==========================================================
    # Heatmap
    # ==========================================================
    hm = sns.heatmap(
        mat_plot,
        ax=ax_main,
        cmap="mako",
        norm=norm,
        xticklabels=False,
        yticklabels=False,
        cbar=True,
        cbar_ax=ax_cbar,
        linewidths=0,
        rasterized=True,
    )

    # Clean background for masked values
    ax_main.set_facecolor("#f0f0f0")


    ax_main.set_xlabel("")
    ax_main.set_ylabel("")

    cbar = hm.collections[0].colorbar
    cbar.set_label("")
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
    cbar.set_label("Saliency", fontsize=22, labelpad=14)
    cbar.ax.tick_params(labelsize=18)
    # ==========================================================
    # 5. Axis cluster bars
    # ==========================================================
    ax_top.imshow(gene_colors[None, :, :], aspect="auto")
    ax_top.set_axis_off()

    ax_left.imshow(pathway_colors[:, None, :], aspect="auto")
    ax_left.set_axis_off()

    # ==========================================================
    # 6. External text labels (clean, non-TeX-bold)
    # ==========================================================
    ax_top.text(
        0.5, 1.9,
        "Genes",
        ha="center",
        va="bottom",
        transform=ax_top.transAxes,
        fontsize=22,
    )

    ax_left.text(
        -1.8, 0.5,
        "Pathways",
        ha="center",
        va="center",
        rotation=90,
        transform=ax_left.transAxes,
        fontsize=22,
    )

    # ==========================================================
    # 7. Alignment + resized colorbar
    # ==========================================================
    fig.canvas.draw()
    main_pos = ax_main.get_position()
    cbar_pos = ax_cbar.get_position()

    # y-axis bar flush
    ax_left.set_position([
        ax_left.get_position().x0,
        main_pos.y0,
        ax_left.get_position().width,
        main_pos.height,
    ])

    # top bar aligned
    ax_top.set_position([
        main_pos.x0,
        ax_top.get_position().y0,
        main_pos.width,
        ax_top.get_position().height,
    ])

    # colorbar: farther, half width & half height
    new_h = main_pos.height * 0.5
    new_w = cbar_pos.width * 0.5

    ax_cbar.set_position([
        cbar_pos.x0 + 0.02,  # ← extra separation from heatmap
        main_pos.y0 + (main_pos.height - new_h) / 2,
        new_w,
        new_h,
    ])

    # ==========================================================
    # 8. Save
    # ==========================================================
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(
        os.path.join(output_path, filename),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def plot_leiden_saliency_heatmap_white_background(
    saliency_pathway_matrix,
    gene_names,
    pathway_names,
    gene_clusters,
    pathway_clusters,
    output_path,
    vmax_percentile=99,
    figsize=(16, 12),
    filename="leiden_gene_pathway_heatmap_axisbars.png",
):

    # ==========================================================
    # 1. Leiden ordering
    # ==========================================================
    gene_order = sorted(
        range(len(gene_names)),
        key=lambda i: gene_clusters[gene_names[i]]
    )
    pathway_order = sorted(
        range(len(pathway_names)),
        key=lambda i: pathway_clusters[pathway_names[i]]
    )

    mat = saliency_pathway_matrix[np.ix_(pathway_order, gene_order)]
    vmax = np.percentile(mat, vmax_percentile)


    # ==========================================================
    # Log-scaled normalization (safe for zeros)
    # ==========================================================
    mat_plot = mat.copy()

    # Mask non-positive values (LogNorm cannot handle ≤ 0)
    mat_plot[mat_plot <= 0] = np.nan

    # Robust limits
    vmin = np.nanpercentile(mat_plot, 3)
    vmax = np.nanpercentile(mat_plot, vmax_percentile)

    # Safety guards
    vmin = max(vmin, 1e-6)
    vmax = max(vmax, vmin * 10)

    norm = LogNorm(vmin=vmin, vmax=vmax)

    # ==========================================================
    # 2. Cluster color bars (fixed palette)
    # ==========================================================
    gene_cluster_ids = [gene_clusters[gene_names[i]] for i in gene_order]
    pathway_cluster_ids = [pathway_clusters[pathway_names[i]] for i in pathway_order]

    def cluster_to_rgb(cid):
        if cid not in CLUSTER_COLORS:
            raise ValueError(f"Cluster ID {cid} not in CLUSTER_COLORS")
        return to_rgb(CLUSTER_COLORS[cid])

    gene_colors = np.array([cluster_to_rgb(c) for c in gene_cluster_ids])
    pathway_colors = np.array([cluster_to_rgb(c) for c in pathway_cluster_ids])


    # ==========================================================
    # 3. Layout
    # ==========================================================
    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        2, 3,
        width_ratios=[0.022, 0.92, 0.058],   # extra space for colorbar
        height_ratios=[0.03, 0.97],
        wspace=0.0,
        hspace=0.0,
    )

    ax_top  = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    # ==========================================================
    # 4. Heatmap
    # ==========================================================
    # hm = sns.heatmap(
    #     mat,
    #     ax=ax_main,
    #     cmap="mako",
    #     vmax=vmax,
    #     xticklabels=False,
    #     yticklabels=False,
    #     cbar=True,
    #     cbar_ax=ax_cbar,
    #     rasterized=True,
    # )
    hm = sns.heatmap(
        mat_plot,
        ax=ax_main,
        cmap="mako",
        norm=norm,
        xticklabels=False,
        yticklabels=False,
        cbar=True,
        cbar_ax=ax_cbar,
        rasterized=True,
    )

    ax_main.set_xlabel("")
    ax_main.set_ylabel("")

    cbar = hm.collections[0].colorbar
    cbar.set_label("")
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
    cbar.set_label("Saliency", fontsize=22, labelpad=14)
    cbar.ax.tick_params(labelsize=18)
    # ==========================================================
    # 5. Axis cluster bars
    # ==========================================================
    ax_top.imshow(gene_colors[None, :, :], aspect="auto")
    ax_top.set_axis_off()

    ax_left.imshow(pathway_colors[:, None, :], aspect="auto")
    ax_left.set_axis_off()

    # ==========================================================
    # 6. External text labels (clean, non-TeX-bold)
    # ==========================================================
    ax_top.text(
        0.5, 1.9,
        "Genes",
        ha="center",
        va="bottom",
        transform=ax_top.transAxes,
        fontsize=22,
    )

    ax_left.text(
        -1.8, 0.5,
        "Pathways",
        ha="center",
        va="center",
        rotation=90,
        transform=ax_left.transAxes,
        fontsize=22,
    )

    # ==========================================================
    # 7. Alignment + resized colorbar
    # ==========================================================
    fig.canvas.draw()
    main_pos = ax_main.get_position()
    cbar_pos = ax_cbar.get_position()

    # y-axis bar flush
    ax_left.set_position([
        ax_left.get_position().x0,
        main_pos.y0,
        ax_left.get_position().width,
        main_pos.height,
    ])

    # top bar aligned
    ax_top.set_position([
        main_pos.x0,
        ax_top.get_position().y0,
        main_pos.width,
        ax_top.get_position().height,
    ])

    # colorbar: farther, half width & half height
    new_h = main_pos.height * 0.5
    new_w = cbar_pos.width * 0.5

    ax_cbar.set_position([
        cbar_pos.x0 + 0.02,  # ← extra separation from heatmap
        main_pos.y0 + (main_pos.height - new_h) / 2,
        new_w,
        new_h,
    ])

    # ==========================================================
    # 8. Save
    # ==========================================================
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(
        os.path.join(output_path, filename),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def plot_leiden_saliency_heatmap_border(
    saliency_pathway_matrix,
    gene_names,
    pathway_names,
    gene_clusters,
    pathway_clusters,
    output_path,
    vmax_percentile=99,
    figsize=(16, 12),
    filename="leiden_gene_pathway_heatmap_axisbars.png",
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import LogNorm, LinearSegmentedColormap, to_rgb
    from matplotlib.ticker import LogFormatterMathtext

    # ==========================================================
    # 0. Dark publication style (global, safe)
    # ==========================================================
    plt.rcParams.update({
        "figure.facecolor": "#0e1117",
        "axes.facecolor":   "#0e1117",
        "savefig.facecolor":"#0e1117",
        "text.color":       "#e6e6e6",
        "axes.labelcolor":  "#e6e6e6",
        "xtick.color":      "#bdbdbd",
        "ytick.color":      "#bdbdbd",
    })

    # ==========================================================
    # 1. Leiden ordering
    # ==========================================================
    gene_order = sorted(
        range(len(gene_names)),
        key=lambda i: gene_clusters[gene_names[i]]
    )
    pathway_order = sorted(
        range(len(pathway_names)),
        key=lambda i: pathway_clusters[pathway_names[i]]
    )

    mat = saliency_pathway_matrix[np.ix_(pathway_order, gene_order)]

    # ==========================================================
    # 2. Log-scaled normalization (robust)
    # ==========================================================
    mat_plot = mat.copy()
    mat_plot[mat_plot <= 0] = np.nan

    vmin = np.nanpercentile(mat_plot, 3)
    vmax = np.nanpercentile(mat_plot, vmax_percentile)

    vmin = max(vmin, 1e-6)
    vmax = max(vmax, vmin * 10)

    norm = LogNorm(vmin=vmin, vmax=vmax)

    # ==========================================================
    # 3. Dark high-contrast colormap (near-black background)
    # ==========================================================
    base_cmap = sns.color_palette("mako", as_cmap=True)
    dark_mako = LinearSegmentedColormap.from_list(
        "dark_mako",
        [
            (0.00, "#050608"),
            (0.15, base_cmap(0.2)),
            (0.40, base_cmap(0.5)),
            (0.70, base_cmap(0.75)),
            (1.00, base_cmap(1.0)),
        ],
    )

    # ==========================================================
    # 4. Cluster color bars
    # ==========================================================
    gene_cluster_ids = [gene_clusters[gene_names[i]] for i in gene_order]
    pathway_cluster_ids = [pathway_clusters[pathway_names[i]] for i in pathway_order]

    def cluster_to_rgb(cid):
        if cid not in CLUSTER_COLORS:
            raise ValueError(f"Cluster ID {cid} not in CLUSTER_COLORS")
        return to_rgb(CLUSTER_COLORS[cid])

    gene_colors = np.array([cluster_to_rgb(c) for c in gene_cluster_ids]) * 0.9
    pathway_colors = np.array([cluster_to_rgb(c) for c in pathway_cluster_ids]) * 0.9

    # ==========================================================
    # 5. Layout
    # ==========================================================
    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        2, 3,
        width_ratios=[0.022, 0.92, 0.058],
        height_ratios=[0.03, 0.97],
        wspace=0.0,
        hspace=0.0,
    )

    ax_top  = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    # ==========================================================
    # 6. Heatmap
    # ==========================================================
    hm = sns.heatmap(
        mat_plot,
        ax=ax_main,
        cmap=dark_mako,
        norm=norm,
        xticklabels=False,
        yticklabels=False,
        cbar=True,
        cbar_ax=ax_cbar,
        rasterized=True,
    )

    ax_main.set_xlabel("")
    ax_main.set_ylabel("")

    # ==========================================================
    # 7. Subtle cluster separators (contrast without clutter)
    # ==========================================================
    prev = pathway_cluster_ids[0]
    for i, cid in enumerate(pathway_cluster_ids):
        if cid != prev:
            ax_main.axhline(i, color="white", lw=0.35, alpha=0.35)
        prev = cid

    prev = gene_cluster_ids[0]
    for i, cid in enumerate(gene_cluster_ids):
        if cid != prev:
            ax_main.axvline(i, color="white", lw=0.35, alpha=0.35)
        prev = cid

    # ==========================================================
    # 8. Colorbar styling
    # ==========================================================
    cbar = hm.collections[0].colorbar
    cbar.set_label("Saliency", fontsize=22, labelpad=14)
    cbar.ax.tick_params(labelsize=18, colors="#e6e6e6")
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
    cbar.outline.set_edgecolor("#e6e6e6")
    cbar.outline.set_linewidth(0.8)
    cbar.ax.set_facecolor("#0e1117")

    # ==========================================================
    # 9. Axis cluster bars
    # ==========================================================
    ax_top.imshow(gene_colors[None, :, :], aspect="auto")
    ax_left.imshow(pathway_colors[:, None, :], aspect="auto")

    ax_top.set_axis_off()
    ax_left.set_axis_off()

    for ax in (ax_top, ax_left):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#1f2933")
            spine.set_linewidth(0.6)

    # ==========================================================
    # 10. Labels
    # ==========================================================
    ax_top.text(
        0.5, 1.9, "Genes",
        ha="center", va="bottom",
        transform=ax_top.transAxes,
        fontsize=22,
    )

    ax_left.text(
        -1.8, 0.5, "Pathways",
        ha="center", va="center",
        rotation=90,
        transform=ax_left.transAxes,
        fontsize=22,
    )

    # ==========================================================
    # 11. Alignment & colorbar resizing
    # ==========================================================
    fig.canvas.draw()
    main_pos = ax_main.get_position()
    cbar_pos = ax_cbar.get_position()

    ax_left.set_position([
        ax_left.get_position().x0,
        main_pos.y0,
        ax_left.get_position().width,
        main_pos.height,
    ])

    ax_top.set_position([
        main_pos.x0,
        ax_top.get_position().y0,
        main_pos.width,
        ax_top.get_position().height,
    ])

    new_h = main_pos.height * 0.5
    new_w = cbar_pos.width * 0.5

    ax_cbar.set_position([
        cbar_pos.x0 + 0.02,
        main_pos.y0 + (main_pos.height - new_h) / 2,
        new_w,
        new_h,
    ])

    # ==========================================================
    # 12. Save
    # ==========================================================
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(
        os.path.join(output_path, filename),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def plot_leiden_saliency_heatmap(
    saliency_pathway_matrix,
    gene_names,
    pathway_names,
    gene_clusters,
    pathway_clusters,
    output_path,
    vmax_percentile=99,
    figsize=(16, 12),
    filename="leiden_gene_pathway_heatmap_grayscale.png",
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    from matplotlib.ticker import LogFormatterMathtext

    # ==========================================================
    # 0. Print-safe grayscale style
    # ==========================================================
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "savefig.facecolor":"white",
        "text.color":       "black",
        "axes.labelcolor":  "black",
        "xtick.color":      "black",
        "ytick.color":      "black",
        "font.size":        14,
    })

    # ==========================================================
    # 1. Leiden ordering
    # ==========================================================
    gene_order = sorted(
        range(len(gene_names)),
        key=lambda i: gene_clusters[gene_names[i]]
    )
    pathway_order = sorted(
        range(len(pathway_names)),
        key=lambda i: pathway_clusters[pathway_names[i]]
    )

    mat = saliency_pathway_matrix[np.ix_(pathway_order, gene_order)]

    # ==========================================================
    # 2. Log-scaled normalization (robust)
    # ==========================================================
    mat_plot = mat.copy()
    mat_plot[mat_plot <= 0] = np.nan

    vmin = np.nanpercentile(mat_plot, 3)
    vmax = np.nanpercentile(mat_plot, vmax_percentile)

    vmin = max(vmin, 1e-6)
    vmax = max(vmax, vmin * 10)

    norm = LogNorm(vmin=vmin, vmax=vmax)

    # ==========================================================
    # 3. High-contrast grayscale colormap
    # ==========================================================
    grayscale_cmap = LinearSegmentedColormap.from_list(
        "print_gray",
        [
            (0.00, "#ffffff"),  # white background
            (0.15, "#e0e0e0"),
            (0.35, "#bdbdbd"),
            (0.60, "#7f7f7f"),
            (0.85, "#404040"),
            (1.00, "#000000"),  # strong signal
        ],
    )

    # ==========================================================
    # 4. Cluster bars (patterned grayscale)
    # ==========================================================
    gene_cluster_ids = [gene_clusters[gene_names[i]] for i in gene_order]
    pathway_cluster_ids = [pathway_clusters[pathway_names[i]] for i in pathway_order]

    def cluster_gray(cid, max_c=20):
        # Deterministic grayscale spacing
        return str(0.15 + 0.7 * ((cid % max_c) / max_c))

    gene_colors = np.array([[float(cluster_gray(c))]*3 for c in gene_cluster_ids])
    pathway_colors = np.array([[float(cluster_gray(c))]*3 for c in pathway_cluster_ids])

    # ==========================================================
    # 5. Layout
    # ==========================================================
    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        2, 3,
        width_ratios=[0.022, 0.92, 0.058],
        height_ratios=[0.03, 0.97],
        wspace=0.0,
        hspace=0.0,
    )

    ax_top  = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    # ==========================================================
    # 6. Heatmap
    # ==========================================================
    hm = sns.heatmap(
        mat_plot,
        ax=ax_main,
        cmap=grayscale_cmap,
        norm=norm,
        xticklabels=False,
        yticklabels=False,
        cbar=True,
        cbar_ax=ax_cbar,
        rasterized=True,
    )

    # ==========================================================
    # 7. Cluster separators (essential for B/W)
    # ==========================================================
    prev = pathway_cluster_ids[0]
    for i, cid in enumerate(pathway_cluster_ids):
        if cid != prev:
            ax_main.axhline(i, color="black", lw=0.5)
        prev = cid

    prev = gene_cluster_ids[0]
    for i, cid in enumerate(gene_cluster_ids):
        if cid != prev:
            ax_main.axvline(i, color="black", lw=0.5)
        prev = cid

    # ==========================================================
    # 8. Colorbar formatting
    # ==========================================================
    cbar = hm.collections[0].colorbar
    cbar.set_label("Saliency", fontsize=18, labelpad=10)
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(1.0)

    # ==========================================================
    # 9. Axis cluster bars
    # ==========================================================
    ax_top.imshow(gene_colors[None, :, :], aspect="auto")
    ax_left.imshow(pathway_colors[:, None, :], aspect="auto")

    ax_top.set_axis_off()
    ax_left.set_axis_off()

    for ax in (ax_top, ax_left):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.6)

    # ==========================================================
    # 10. Labels
    # ==========================================================
    ax_top.text(
        0.5, 1.9, "Genes",
        ha="center", va="bottom",
        transform=ax_top.transAxes,
        fontsize=18,
    )

    ax_left.text(
        -1.8, 0.5, "Pathways",
        ha="center", va="center",
        rotation=90,
        transform=ax_left.transAxes,
        fontsize=18,
    )

    # ==========================================================
    # 11. Alignment
    # ==========================================================
    fig.canvas.draw()
    main_pos = ax_main.get_position()
    cbar_pos = ax_cbar.get_position()

    ax_left.set_position([
        ax_left.get_position().x0,
        main_pos.y0,
        ax_left.get_position().width,
        main_pos.height,
    ])

    ax_top.set_position([
        main_pos.x0,
        ax_top.get_position().y0,
        main_pos.width,
        ax_top.get_position().height,
    ])

    new_h = main_pos.height * 0.5
    new_w = cbar_pos.width * 0.5

    ax_cbar.set_position([
        cbar_pos.x0 + 0.02,
        main_pos.y0 + (main_pos.height - new_h) / 2,
        new_w,
        new_h,
    ])

    # ==========================================================
    # 12. Save
    # ==========================================================
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(
        os.path.join(output_path, filename),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def cluster_centers(order, names, clusters):
    cluster_indices = defaultdict(list)
    for pos, idx in enumerate(order):
        cluster_indices[clusters[names[idx]]].append(pos)

    centers = {
        c: int(np.mean(v)) for c, v in cluster_indices.items()
    }
    return centers



def leiden_bipartite_from_saliency(
    saliency_pathway_matrix,
    gene_names,
    pathway_names,
    resolution=1.2,
    weight_threshold=0.0
):
    n_pathways, n_genes = saliency_pathway_matrix.shape
    edges = []
    weights = []

    for p in range(n_pathways):
        row = saliency_pathway_matrix[p]
        nz = np.where(row > weight_threshold)[0]
        for g in nz:
            edges.append((p, n_pathways + g))
            weights.append(float(row[g]))

    g = ig.Graph(
        n=n_pathways + n_genes,
        edges=edges,
        edge_attrs={"weight": weights}
    )

    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution
    )

    labels = np.array(part.membership)

    pathway_clusters = {
        pathway_names[i]: int(labels[i])
        for i in range(n_pathways)
    }

    gene_clusters = {
        gene_names[j]: int(labels[n_pathways + j])
        for j in range(n_genes)
    }

    return pathway_clusters, gene_clusters, labels

def plot_gene_umap(
    gene_embeddings,
    gene_names,
    gene_clusters,
    output_path
):
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=0
    )

    emb_2d = reducer.fit_transform(gene_embeddings)

    labels = np.array([gene_clusters[g] for g in gene_names])

    plt.figure(figsize=(7, 6))
    plt.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=labels,
        s=8,
        cmap="tab20"
    )

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "gene_umap_leiden.png"),
        dpi=300
    )
    plt.close()

def plot_top_genes_per_cluster(
    gene_saliency,
    gene_names,
    gene_clusters,
    output_path,
    top_k=10
):
    common_genes = [g for g in gene_names if g in gene_clusters]
    saliency_filtered = [gene_saliency[gene_names.index(g)] for g in common_genes]

    df = pd.DataFrame({
        "gene": common_genes,
        "saliency": saliency_filtered,
        "cluster": [gene_clusters[g] for g in common_genes]
    })


    for c in sorted(df["cluster"].unique()):
        top = df[df["cluster"] == c].nlargest(top_k, "saliency")

        plt.figure(figsize=(4, 3))
        plt.barh(top["gene"], top["saliency"])
        plt.title(f"Gene Cluster {c}")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        plt.savefig(
            os.path.join(output_path, f"top_genes_cluster_{c}.png"),
            dpi=300
        )
        plt.close()

def build_gene_cluster_map(gene_clusters):
    m = defaultdict(list)
    for g, c in gene_clusters.items():
        m[c].append(g)
    return dict(m)





def aggregate_pathway_saliency_from_enrichment(
    gene_saliency: np.ndarray,
    enrich_matrix: np.ndarray,
    normalize: bool = True
):
    """
    Aggregate pathway-level saliency using enrichment-weighted gene importance.

    Args:
        gene_saliency: [G] array, saliency per gene
        enrich_matrix: [G × P] enrichment score matrix
        normalize: whether to normalize pathway saliency

    Returns:
        pathway_saliency: [P] array
    """
    pathway_saliency = gene_saliency @ enrich_matrix  # (G)ᵀ × (G×P) → (P)

    if normalize:
        pathway_saliency = pathway_saliency / (
            pathway_saliency.sum() + 1e-9
        )

    return pathway_saliency


def aggregate_pathway_saliency(
    relevance,
    G_dgl,
    pathway_names,
    pathway_key="pathway_name",
    reduction="mean"
):
    """
    Aggregate node-level saliency into pathway-level saliency.

    Parameters
    ----------
    relevance : torch.Tensor
        Shape (num_nodes, num_features)
    G_dgl : dgl.DGLGraph
        Graph used for training; must contain pathway IDs per node
    pathway_names : list[str]
        Ordered list of unique pathway names
    pathway_key : str
        Node data key storing pathway identity
    reduction : str
        "mean" or "sum"

    Returns
    -------
    pathway_saliency : np.ndarray
        Shape (num_pathways,)
    """

    if pathway_key not in G_dgl.ndata:
        raise KeyError(
            f"G_dgl.ndata['{pathway_key}'] not found — cannot aggregate saliency"
        )

    node_saliency = relevance.sum(dim=1).detach().cpu().numpy()
    node_pathways = G_dgl.ndata[pathway_key]

    if torch.is_tensor(node_pathways):
        node_pathways = node_pathways.detach().cpu().numpy()

    pathway_to_idx = {p: i for i, p in enumerate(pathway_names)}

    pathway_saliency = np.zeros(len(pathway_names), dtype=np.float64)
    counts = np.zeros(len(pathway_names), dtype=np.int64)

    for s, p in zip(node_saliency, node_pathways):
        if p in pathway_to_idx:
            idx = pathway_to_idx[p]
            pathway_saliency[idx] += s
            counts[idx] += 1

    if reduction == "mean":
        pathway_saliency /= np.maximum(counts, 1)

    return pathway_saliency


def plot_gene_umap(embeddings, gene_labels, output_path):
    import umap
    import matplotlib.pyplot as plt

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    Z = reducer.fit_transform(embeddings)

    cmap = {
        "Driver": "#d62728",
        "NonDriver": "#1f77b4",
        "Unknown": "#7f7f7f"
    }

    plt.figure(figsize=(7, 6))
    plt.scatter(
        Z[:, 0],
        Z[:, 1],
        c=[cmap[l] for l in gene_labels],
        s=10,
        alpha=0.8
    )
    for k, v in cmap.items():
        plt.scatter([], [], c=v, label=k, s=30)
    plt.legend(frameon=False)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"{output_path}/umap_genes_driver_colored.png", dpi=300)
    plt.close()

def build_saliency_pathway_matrix(
    saliency_np,
    enrich_matrix,
    gene_to_idx,
    pathway_to_idx
):
    import numpy as np

    M = np.zeros((len(pathway_to_idx), len(gene_to_idx)))
    for g, gi in gene_to_idx.items():
        for p, pi in pathway_to_idx.items():
            M[pi, gi] = saliency_np[gi] * enrich_matrix[gi, pi]
    return M


def run_biclustering(saliency_pathway_matrix, output_path):
    import os
    import numpy as np
    from sklearn.cluster import SpectralBiclustering

    model = SpectralBiclustering(
        n_clusters=(10, 10),
        method="log",
        random_state=0
    )
    model.fit(saliency_pathway_matrix)

    np.save(
        os.path.join(output_path, "bicluster_row_labels.npy"),
        model.row_labels_
    )

    return model.row_labels_, model.column_labels_


def save_top_gene_pathway_pairs(
    saliency_pathway_matrix,
    gene_names,
    pathway_names,
    output_path,
    top_k=50
):
    import os
    import numpy as np
    import pandas as pd

    rows = []
    for p in range(len(pathway_names)):
        idx = np.argsort(saliency_pathway_matrix[p])[::-1][:top_k]
        for g in idx:
            rows.append([
                pathway_names[p],
                gene_names[g],
                saliency_pathway_matrix[p, g]
            ])

    pd.DataFrame(
        rows,
        columns=["Pathway", "Gene", "Saliency"]
    ).to_csv(
        os.path.join(output_path, "top_gene_pathway_pairs.csv"),
        index=False
    )


def save_cluster_pathway_gene_flows(
    saliency_pathway_matrix,
    row_labels,
    gene_names,
    pathway_names,
    output_path,
    top_k=50
):
    import os
    import numpy as np
    import pandas as pd

    flows = []
    for p, c in enumerate(row_labels):
        scores = saliency_pathway_matrix[p]
        idx = np.argsort(scores)[::-1][:top_k]
        for g in idx:
            if scores[g] > 0:
                flows.append([
                    f"Cluster {c}",
                    pathway_names[p],
                    gene_names[g],
                    scores[g]
                ])

    pd.DataFrame(
        flows,
        columns=["Cluster", "Pathway", "Gene", "Value"]
    ).to_csv(
        os.path.join(output_path, "sankey_cluster_pathway_gene.csv"),
        index=False
    )


def assign_gene_clusters(gene_names, col_labels):
    import pandas as pd
    return pd.Series(col_labels, index=gene_names)


def align_expression_matrix(expr_path, gene_names):
    import pandas as pd

    expr_df = pd.read_csv(expr_path, sep="\t", index_col=0)
    common = sorted(set(expr_df.index).intersection(gene_names))
    return expr_df.loc[common].T.to_numpy(), common


def extract_gene_cluster_map(
    saliency_pathway_matrix,
    row_labels,
    top_k=50
):
    import numpy as np
    from collections import defaultdict

    m = defaultdict(set)
    for p, c in enumerate(row_labels):
        idx = np.argsort(saliency_pathway_matrix[p])[::-1][:top_k]
        for g in idx:
            m[c].add(g)
    return {k: list(v) for k, v in m.items()}



def plot_joint_gene_pathway_umap_x(enrich_matrix, saliency_pathway, gene_labels, output_path):
    import umap
    import numpy as np
    import matplotlib.pyplot as plt

    X = np.vstack([enrich_matrix, saliency_pathway])
    labels = gene_labels + ["Pathway"] * saliency_pathway.shape[0]

    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )
    Z = reducer.fit_transform(X)

    gene_idx = np.arange(len(gene_labels))
    path_idx = np.arange(len(gene_labels), Z.shape[0])

    cmap = {
        "Driver": "#d62728",
        "NonDriver": "#1f77b4",
        "Unknown": "#7f7f7f",
        "Pathway": "#2ca02c"
    }

    plt.figure(figsize=(8, 7))
    plt.scatter(
        Z[gene_idx, 0],
        Z[gene_idx, 1],
        c=[cmap[l] for l in labels[:len(gene_idx)]],
        s=12,
        alpha=0.75
    )
    plt.scatter(
        Z[path_idx, 0],
        Z[path_idx, 1],
        c=[cmap["Pathway"]] * len(path_idx),
        s=60,
        marker="^",
        edgecolors="k",
        linewidths=0.3
    )
    for k, v in cmap.items():
        plt.scatter([], [], c=v, label=k, s=40)
    plt.legend(frameon=False)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"{output_path}/joint_gene_pathway_umap.png", dpi=300)
    plt.close()

def compute_relevance_scores(model, graph, features, node_indices=None, method="saliency", use_abs=True, baseline=None, steps=50):
    """
    Computes relevance scores for selected nodes using either saliency (gradients) or integrated gradients (IG).

    Args:
        model: Trained GNN model
        graph: DGL graph
        features: Input node features (torch.Tensor or np.ndarray)
        node_indices: List/Tensor of node indices to compute relevance for. If None, auto-select using probs > 0.0
        method: "saliency" or "integrated_gradients"
        use_abs: Whether to use absolute values of gradients
        baseline: Baseline input for IG (default: zero vector)
        steps: Number of steps for IG approximation

    Returns:
        relevance_scores: Tensor of shape [num_nodes, num_features] (0s for nodes not analyzed)
    """
    model.eval()
    if isinstance(features, np.ndarray):
        features = torch.tensor(features, dtype=torch.float32)

    features = features.clone().detach().requires_grad_(True)

    with torch.enable_grad():
        logits = model(graph, features)
        probs = torch.sigmoid(logits.squeeze())

        if node_indices is None:
            node_indices = torch.nonzero(probs > 0.0, as_tuple=False).squeeze()
            if node_indices.ndim == 0:
                node_indices = node_indices.unsqueeze(0)

        relevance_scores = torch.zeros_like(features)

        for i, idx in enumerate(tqdm(node_indices, desc=f"Computing relevance ({method})", leave=True)):
            model.zero_grad()
            if features.grad is not None:
                features.grad.zero_()

            if method == "saliency":
                probs[idx].backward(retain_graph=(i != len(node_indices) - 1))
                grads = features.grad[idx]
                relevance_scores[idx] = grads.abs().detach() if use_abs else grads.detach()

            elif method == "integrated_gradients":
                # Define baseline
                if baseline is None:
                    baseline_input = torch.zeros_like(features)
                else:
                    baseline_input = baseline.clone().detach()

                # Generate scaled inputs
                total_grad = torch.zeros_like(features)
                for alpha in range(1, steps + 1):
                    scaled_input = baseline_input + (alpha / steps) * (features - baseline_input)
                    scaled_input.requires_grad_()

                    out = model(graph, scaled_input)
                    prob = torch.sigmoid(out.squeeze())[idx]

                    model.zero_grad()
                    if scaled_input.grad is not None:
                        scaled_input.grad.zero_()

                    prob.backward(retain_graph=True)
                    grad = scaled_input.grad
                    total_grad += grad

                avg_grad = total_grad / steps
                ig = (features - baseline_input) * avg_grad
                relevance_scores[idx] = ig[idx].abs() if use_abs else ig[idx]

            else:
                raise ValueError(f"Unknown method: {method}. Use 'saliency' or 'integrated_gradients'.")

    return relevance_scores

def plot_km(df, cluster, output_path):
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(5, 4))

    for grp in ["High", "Low"]:
        mask = df[f"{cluster}_group"] == grp
        kmf.fit(
            df.loc[mask, "time"],
            df.loc[mask, "event"],
            label=f"{cluster} {grp}"
        )
        kmf.plot_survival_function(ci_show=False)

    g1 = df[f"{cluster}_group"] == "High"
    g2 = df[f"{cluster}_group"] == "Low"

    res = logrank_test(
        df.loc[g1, "time"],
        df.loc[g2, "time"],
        df.loc[g1, "event"],
        df.loc[g2, "event"]
    )

    # ----------------------------
    # p-value bottom-left (axes coords)
    # ----------------------------
    plt.text(
        0.02, 0.02,
        f"Log-rank p = {res.p_value:.2e}",
        transform=plt.gca().transAxes,
        ha="left",
        va="bottom",
        fontsize=9
    )

    plt.title(cluster)
    plt.xlabel("Days")
    plt.ylabel("Overall Survival")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, f"KM_{cluster}.pdf")
    )
    plt.close()

def decision_curve_analysis(event, risk, thresholds):
    """
    event: binary array (0/1)
    risk: continuous risk score (higher = worse)
    """
    N = len(event)
    nb = []

    for pt in thresholds:
        preds = risk >= pt

        TP = np.sum((preds == 1) & (event == 1))
        FP = np.sum((preds == 1) & (event == 0))

        net_benefit = (TP / N) - (FP / N) * (pt / (1 - pt))
        nb.append(net_benefit)

    return np.array(nb)

def get_pvalue_column(df):
    candidates = [
        "p_value",
        "pvalue",
        "P-value",
        "p.adjust",
        "adj_p_value",
        "FDR",
        "q_value"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"No p-value column found. Available columns: {df.columns.tolist()}"
    )

def plot_pathway_enrichment_dotplot(
    df_enrich,
    output_path, 
    top_k=10
):
    df = df_enrich.copy()
    pcol = get_pvalue_column(df)

    df["-log10(p)"] = -np.log10(df[pcol])
    df = df.sort_values(pcol)


    # df["-log10(p)"] = -np.log10(df["p_value"])

    # keep top pathways per cluster
    df = (
        df.sort_values("pvalue")
          .groupby("Cluster")
          .head(top_k)
    )

    plt.figure(figsize=(7, 0.4 * df.shape[0]))

    sns.scatterplot(
        data=df,
        x="gene_ratio",
        y="Pathway",
        size="gene_count",
        hue="-log10(p)",
        palette="viridis",
        sizes=(40, 300),
        legend="brief"
    )

    plt.xlabel("Gene Ratio")
    plt.ylabel("")
    plt.title("Pathway Enrichment per Bicluster")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "pathway_enrichment_dotplot.pdf")
    )
    plt.close()

def plot_patient_bicluster_heatmap(
    patient_cluster_scores,
    patient_bicluster,
    output_path
):
    ordered_patients = (
        patient_bicluster
        .loc[patient_cluster_scores.index]
        .sort_values()
        .index
    )

    ordered_clusters = (
        patient_cluster_scores
        .mean(axis=0)
        .sort_values(ascending=False)
        .index
    )

    data = patient_cluster_scores.loc[
        ordered_patients, ordered_clusters
    ]

    plt.figure(figsize=(6, 8))
    sns.heatmap(
        data,
        cmap="RdBu_r",
        center=0,
        yticklabels=False,
        xticklabels=True,
        cbar_kws={"label": "Mean expression (risk score)"}
    )

    plt.xlabel("Bicluster")
    plt.ylabel("Patients (ordered by bicluster)")
    plt.title("Patient × Bicluster Risk Heatmap")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "patient_bicluster_heatmap.pdf")
    )
    plt.close()

def build_and_plot_gene_pathway_modules_centered(
    df_enrich,
    output_path,
    top_genes_per_pathway=15,
    top_pathways=4,
    pval_cutoff=0.05
):
    """
    Plot top pathways with genes radially arranged around each pathway.
    """

    os.makedirs(output_path, exist_ok=True)

    # ----------------------------
    # 1. FILTER SIGNIFICANT
    # ----------------------------
    df = df_enrich.copy()
    df = df[df["pvalue"] <= pval_cutoff]
    df["enrich_score"] = -np.log10(df["pvalue"])

    # ----------------------------
    # 2. SELECT TOP PATHWAYS
    # ----------------------------
    top_pathway_list = (
        df.groupby("PathwayB")["enrich_score"]
        .sum()
        .sort_values(ascending=False)
        .head(top_pathways)
        .index.tolist()
    )

    # ----------------------------
    # 3. FIGURE SETUP
    # ----------------------------
    ncols = 2
    nrows = int(np.ceil(len(top_pathway_list) / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 6 * nrows)
    )

    axes = np.array(axes).reshape(-1)

    # ----------------------------
    # 4. PLOT EACH PATHWAY MODULE
    # ----------------------------
    for ax, pathway in zip(axes, top_pathway_list):

        df_p = df[df["PathwayB"] == pathway]

        top_genes = (
            df_p.groupby("Gene2")["enrich_score"]
            .sum()
            .sort_values(ascending=False)
            .head(top_genes_per_pathway)
            .index.tolist()
        )

        df_p = df_p[df_p["Gene2"].isin(top_genes)]

        # Build graph
        G = nx.Graph()
        G.add_node(pathway, node_type="pathway")

        for _, row in df_p.iterrows():
            G.add_node(row["Gene2"], node_type="gene")
            G.add_edge(
                pathway,
                row["Gene2"],
                weight=row["enrich_score"]
            )

        # Layout: pathway center, genes in circle
        pos = {pathway: (0, 0)}
        angles = np.linspace(0, 2 * np.pi, len(top_genes), endpoint=False)

        for angle, gene in zip(angles, top_genes):
            pos[gene] = (np.cos(angle), np.sin(angle))

        # Draw
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[pathway],
            node_color="#a6cee3",
            node_size=1200,
            ax=ax
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=top_genes,
            node_color="#b2df8a",
            node_size=500,
            ax=ax
        )

        nx.draw_networkx_edges(
            G,
            pos,
            width=[
                1 + G[u][v]["weight"] * 0.15
                for u, v in G.edges()
            ],
            edge_color="gray",
            alpha=0.75,
            ax=ax
        )

        nx.draw_networkx_labels(
            G,
            pos,
            font_size=12,
            ax=ax
        )

        ax.set_title(pathway, fontsize=11, fontweight="bold")
        ax.axis("off")

    # Remove unused axes
    for ax in axes[len(top_pathway_list):]:
        ax.axis("off")

    plt.tight_layout()

    # ----------------------------
    # 5. SAVE
    # ----------------------------
    plt.savefig(
        os.path.join(output_path, "top4_pathway_gene_modules.png"),
        dpi=300
    )
    plt.savefig(
        os.path.join(output_path, "top4_pathway_gene_modules.pdf")
    )
    plt.close()

    print(f"[✓] Saved top-{top_pathways} pathway-centered gene modules")

def edge_integrated_gradients_no_tqdm(
    h,
    u_idx,
    v_idx,
    predictor,
    steps=50
):
    """
    h        : node embeddings (N × d)
    u_idx,v_idx : edge endpoints
    predictor   : trained MLPPredictor
    """
    device = h.device

    # Edge embedding
    edge_embed = torch.cat([h[u_idx], h[v_idx]], dim=1)
    baseline = torch.zeros_like(edge_embed)

    total_grad = torch.zeros_like(edge_embed)

    for alpha in torch.linspace(0, 1, steps, device=device):
        interp = baseline + alpha * (edge_embed - baseline)
        interp.requires_grad_(True)

        score = predictor.forward_from_embedding(interp).sum()
        score.backward()

        total_grad += interp.grad.detach()

    avg_grad = total_grad / steps
    ig = (edge_embed - baseline) * avg_grad

    # scalar attribution per edge
    return ig.abs().sum(dim=1)

def build_and_plot_gene_pathway_umap(
    df_enrich,
    output_path,
    top_genes=40,
    top_pathways=40,
    pval_cutoff=0.05,
    random_state=42
):
    """
    Gene–pathway bipartite visualization using UMAP layout.
    """

    os.makedirs(output_path, exist_ok=True)

    # ----------------------------
    # 1. FILTER SIGNIFICANT
    # ----------------------------
    df = df_enrich.copy()
    df = df[df["pvalue"] <= pval_cutoff]

    df["enrich_score"] = -np.log10(df["pvalue"])

    # ----------------------------
    # 2. SELECT TOP NODES
    # ----------------------------
    top_gene_list = (
        df.groupby("Gene2")["enrich_score"]
        .sum()
        .sort_values(ascending=False)
        .head(top_genes)
        .index.tolist()
    )

    top_pathway_list = (
        df.groupby("PathwayB")["enrich_score"]
        .sum()
        .sort_values(ascending=False)
        .head(top_pathways)
        .index.tolist()
    )

    df = df[
        df["Gene2"].isin(top_gene_list) &
        df["PathwayB"].isin(top_pathway_list)
    ]

    # ----------------------------
    # 3. BUILD BIPARTITE GRAPH
    # ----------------------------
    G = nx.Graph()

    for g in top_gene_list:
        G.add_node(g, node_type="gene")

    for p in top_pathway_list:
        G.add_node(p, node_type="pathway")

    for _, row in df.iterrows():
        G.add_edge(
            row["Gene2"],
            row["PathwayB"],
            weight=row["enrich_score"]
        )

    # ----------------------------
    # 4. UMAP LAYOUT (NODE EMBEDDING)
    # ----------------------------
    nodes = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes)

    reducer = umap.UMAP(
        n_neighbors=10,
        min_dist=0.3,
        metric="cosine",
        random_state=random_state
    )

    emb = reducer.fit_transform(A)

    pos = {node: emb[i] for i, node in enumerate(nodes)}

    # ----------------------------
    # 5. PLOT
    # ----------------------------
    fig, ax = plt.subplots(figsize=(9, 8))

    genes = [n for n, d in G.nodes(data=True) if d["node_type"] == "gene"]
    pathways = [n for n, d in G.nodes(data=True) if d["node_type"] == "pathway"]

    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.25,
        width=1.0,
        ax=ax
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=genes,
        node_color="#1f78b4",
        node_size=200,
        label="Genes",
        ax=ax
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=pathways,
        node_color="#33a02c",
        node_size=350,
        label="Pathways",
        ax=ax
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=12,
        ax=ax
    )

    ax.set_title("Gene–Pathway Modules (UMAP layout)")
    ax.axis("off")
    ax.legend(loc="best")

    plt.tight_layout()

    # ----------------------------
    # 6. SAVE
    # ----------------------------
    plt.savefig(
        os.path.join(output_path, "gene_pathway_umap.png"),
        dpi=300
    )
    plt.savefig(
        os.path.join(output_path, "gene_pathway_umap.pdf")
    )
    plt.close()

    nx.write_graphml(
        G,
        os.path.join(output_path, "gene_pathway_umap.graphml")
    )

    print(
        f"[✓] UMAP graph saved with "
        f"{len(genes)} genes and {len(pathways)} pathways"
    )

def build_and_plot_gene_pathway_modules(
    df_enrich,
    output_path,
    top_genes=30,
    top_pathways=30,
    pval_cutoff=0.05
):
    """
    Builds and plots gene–pathway bipartite modules safely.
    """

    os.makedirs(output_path, exist_ok=True)

    # ----------------------------
    # 1. FILTER SIGNIFICANT
    # ----------------------------
    df = df_enrich.copy()
    df = df[df["pvalue"] <= pval_cutoff]

    df["enrich_score"] = -np.log10(df["pvalue"])

    # ----------------------------
    # 2. TOP GENES / PATHWAYS
    # ----------------------------
    top_gene_list = (
        df.groupby("Gene2")["enrich_score"]
        .sum()
        .sort_values(ascending=False)
        .head(top_genes)
        .index.tolist()
    )

    top_pathway_list = (
        df.groupby("PathwayB")["enrich_score"]
        .sum()
        .sort_values(ascending=False)
        .head(top_pathways)
        .index.tolist()
    )

    df = df[
        df["Gene2"].isin(top_gene_list) &
        df["PathwayB"].isin(top_pathway_list)
    ]

    # ----------------------------
    # 3. BUILD BIPARTITE GRAPH
    # ----------------------------
    G = nx.Graph()

    for g in top_gene_list:
        G.add_node(g, bipartite="gene")

    for p in top_pathway_list:
        G.add_node(p, bipartite="pathway")

    for _, row in df.iterrows():
        G.add_edge(
            row["Gene2"],
            row["PathwayB"],
            weight=row["enrich_score"]
        )

    # ----------------------------
    # 4. LAYOUT (SAFE SIZE)
    # ----------------------------
    genes = [n for n, d in G.nodes(data=True) if d["bipartite"] == "gene"]
    pathways = [n for n, d in G.nodes(data=True) if d["bipartite"] == "pathway"]

    pos = {}
    pos.update((g, (0, i)) for i, g in enumerate(genes))
    pos.update((p, (1, i)) for i, p in enumerate(pathways))

    # ----------------------------
    # 5. PLOT
    # ----------------------------
    fig_h = max(6, 0.25 * max(len(genes), len(pathways)))
    fig, ax = plt.subplots(figsize=(10, fig_h))

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=genes,
        node_color="#1f78b4",
        node_size=300,
        label="Genes",
        ax=ax
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=pathways,
        node_color="#33a02c",
        node_size=400,
        label="Pathways",
        ax=ax
    )

    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.5,
        width=1.2,
        ax=ax
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=12,
        ax=ax
    )

    ax.set_title("Gene–Pathway Bipartite Module", fontsize=12)
    ax.axis("off")
    ax.legend(loc="upper center", ncol=2)

    plt.tight_layout()

    # ----------------------------
    # 6. SAVE
    # ----------------------------
    plt.savefig(
        os.path.join(output_path, "gene_pathway_module.png"),
        dpi=300
    )
    plt.savefig(
        os.path.join(output_path, "gene_pathway_module.pdf")
    )
    plt.close()

    nx.write_graphml(
        G,
        os.path.join(output_path, "gene_pathway_module.graphml")
    )

    print(
        f"[✓] Saved gene–pathway module with "
        f"{len(genes)} genes and {len(pathways)} pathways"
    )

def plot_gene_pathway_bipartite(
    G,
    module_id,
    output_path,
    max_genes=30,
    max_pathways=15,
    seed=42
):
    """
    Visualize gene–pathway bipartite graph
    """

    os.makedirs(output_path, exist_ok=True)

    # Separate node sets
    genes = [n for n, d in G.nodes(data=True) if d["bipartite"] == "gene"]
    pathways = [n for n, d in G.nodes(data=True) if d["bipartite"] == "pathway"]

    # Subsample for readability
    genes = genes[:max_genes]
    pathways = pathways[:max_pathways]

    G = G.subgraph(genes + pathways)

    # Layout
    pos = {}
    pos.update(
        (n, (0, i))
        for i, n in enumerate(genes)
    )
    pos.update(
        (n, (1, i))
        for i, n in enumerate(pathways)
    )

    plt.figure(figsize=(10, max(len(genes), len(pathways)) * 0.25))

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.4,
        width=1
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=genes,
        node_color="#4daf4a",
        node_size=200,
        label="Genes"
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=pathways,
        node_color="#377eb8",
        node_size=500,
        label="Pathways"
    )

    # Labels
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=12
    )

    plt.title(f"Gene–Pathway Module {module_id}")
    plt.axis("off")
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_path,
            f"gene_pathway_module_{module_id}.pdf"
        )
    )
    plt.close()

def build_gene_pathway_module_graph(
    gene_clusters,
    df_enrich,
    module_id,
    pval_thresh=0.05
):
    """
    Returns a NetworkX bipartite graph for a single gene–pathway module
    """

    pcol = get_pvalue_column(df_enrich)

    # Genes in module
    genes = gene_clusters[gene_clusters == module_id].index.tolist()

    # Pathways enriched for module
    df_mod = df_enrich[
        (df_enrich["cluster"] == module_id) &
        (df_enrich[pcol] < pval_thresh)
    ]

    pathways = df_mod["pathway"].unique().tolist()

    # Create bipartite graph
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(genes, bipartite="gene")
    G.add_nodes_from(pathways, bipartite="pathway")

    # Add edges (gene → pathway)
    for _, row in df_mod.iterrows():
        for g in genes:
            G.add_edge(g, row["pathway"])

    return G

def get_pvalue_column(df):
    for c in [
        "p_value", "p.adjust", "pvalue",
        "P-value", "FDR", "q_value"
    ]:
        if c in df.columns:
            return c
    raise ValueError(
        f"No p-value column found. Available columns: {df.columns.tolist()}"
    )

def get_pvalue_column(df):
    for c in [
        "p_value", "p.adjust", "pvalue",
        "P-value", "FDR", "q_value"
    ]:
        if c in df.columns:
            return c
    raise ValueError(
        f"No p-value column found. Available columns: {df.columns.tolist()}"
    )

def add_high_low_groups(df_surv):
    for c in df_surv.columns.drop(["time", "event"]):
        df_surv[f"{c}_group"] = (
            df_surv[c] >= df_surv[c].median()
        ).map({True: "High", False: "Low"})
    return df_surv

def plot_patient_cluster_heatmap(
    patient_cluster_scores,
    surv,
    output_path
):
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt

    ordered = (
        patient_cluster_scores
        .mean(axis=1)
        .sort_values(ascending=False)
        .index
    )

    X = patient_cluster_scores.loc[ordered]
    X = (X - X.mean()) / X.std()

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        X,
        cmap="vlag",
        center=0,
        yticklabels=False,
        cbar_kws={"label": "Z-score"}
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_path,
            "heatmap_patient_cluster_bicluster_aligned.pdf"
        )
    )
    plt.close()

def evaluate_survival(df_surv, output_path):

    y = Surv.from_arrays(
        event=df_surv["event"].astype(bool).values,
        time=df_surv["time"].values
    )

    for c in df_surv.columns.drop(["time", "event"]):
        risk = df_surv[c].values

        ci_ipcw = concordance_index_ipcw(y, y, -risk)[0]
        ci_h = concordance_index(df_surv["time"], -risk, df_surv["event"])

        times = np.linspace(
            df_surv["time"].quantile(0.05),
            df_surv["time"].quantile(0.95),
            10
        )

        auc, mean_auc = cumulative_dynamic_auc(y, y, -risk, times)

        plt.figure(figsize=(5, 4))
        plt.plot(times / 365.0, auc, marker="o")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_path, f"ROC_time_{c}.pdf")
        )
        plt.close()

        print(
            f"{c} | IPCW: {ci_ipcw:.3f} | "
            f"Harrell: {ci_h:.3f} | AUC: {mean_auc:.3f}"
        )

def compute_patient_cluster_scores(
    saliency_pathway_matrix,
    row_labels,
    gene_names,
    gene_to_expr_idx,
    expr_matrix,
    top_k=50
):
    import numpy as np
    import pandas as pd
    from collections import defaultdict

    gene_cluster_map = defaultdict(set)

    for p, c in enumerate(row_labels):
        idx = np.argsort(saliency_pathway_matrix[p])[::-1][:top_k]
        for g in idx:
            gene = gene_names[g]
            if gene in gene_to_expr_idx:
                gene_cluster_map[c].add(gene_to_expr_idx[gene])

    scores = {}
    for c, genes in gene_cluster_map.items():
        if len(genes) > 0:
            scores[f"Cluster_{c}"] = expr_matrix[:, list(genes)].mean(axis=1)

    return pd.DataFrame(scores)

def edge_integrated_gradients_cached(
    h,
    u_idx,
    v_idx,
    predictor,
    output_path,
    steps=50,
    batch_size=4096,
    force_recompute=False
):
    os.makedirs(output_path, exist_ok=True)
    cache_file = os.path.join(output_path, "edge_ig.npy")

    if os.path.exists(cache_file) and not force_recompute:
        print("✅ Loading cached edge integrated gradients")
        return torch.from_numpy(np.load(cache_file))

    device = h.device
    E = u_idx.shape[0]
    ig_scores = []

    for i in tqdm(
        range(0, E, batch_size),
        desc="Edge Integrated Gradients (batched)",
        leave=True
    ):
        ub = u_idx[i:i + batch_size]
        vb = v_idx[i:i + batch_size]

        edge_embed = torch.cat([h[ub], h[vb]], dim=1)
        baseline = torch.zeros_like(edge_embed)
        total_grad = torch.zeros_like(edge_embed)

        alphas = torch.linspace(0, 1, steps, device=device)

        for alpha in alphas:
            interp = baseline + alpha * (edge_embed - baseline)
            interp.requires_grad_(True)

            predictor.zero_grad(set_to_none=True)
            predictor.forward_from_embedding(interp).sum().backward()

            total_grad += interp.grad.detach()

        avg_grad = total_grad / steps
        ig = (edge_embed - baseline) * avg_grad
        ig_scores.append(ig.abs().sum(dim=1))

    ig_scores = torch.cat(ig_scores).cpu().numpy()
    np.save(cache_file, ig_scores)

    print(f"✅ Saved edge IG → {cache_file}")

    return torch.from_numpy(ig_scores)

def preprocess_expression(expr_path, surv, gene_names):
    import pandas as pd

    # Load expression
    expr_df = pd.read_csv(
        expr_path,
        sep="\t",
        index_col=0
    )

    # sample → patient
    expr_df.columns = [c[:12] for c in expr_df.columns]

    # Align patients
    patient_ids = sorted(
        set(expr_df.columns).intersection(surv.index)
    )

    expr_df = expr_df[patient_ids]
    surv = surv.loc[patient_ids]

    # Align genes
    common_genes = sorted(set(expr_df.index).intersection(gene_names))
    expr_df = expr_df.loc[common_genes]

    gene_to_expr_idx = {g: i for i, g in enumerate(common_genes)}

    expr_matrix = expr_df.T.to_numpy()  # patients × genes

    return expr_matrix, expr_df, surv, patient_ids, common_genes, gene_to_expr_idx

def load_survival(path):
    import pandas as pd

    surv = pd.read_csv(path, sep="\t")
    if "_PATIENT" in surv.columns:
        surv = surv.rename(columns={"_PATIENT": "patient_id"})
    if "OS.time" in surv.columns:
        surv = surv.rename(columns={"OS.time": "time"})
    if "OS" in surv.columns:
        surv = surv.rename(columns={"OS": "event"})
    surv = surv.set_index("patient_id")[["time", "event"]]
    surv["time"] = surv["time"].astype(float)
    surv["event"] = surv["event"].astype(int)
    return surv


def km_pathway_family(df_surv, patient_family_scores, output_path):
    import os
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    km_dir = os.path.join(output_path, "KM_family")
    os.makedirs(km_dir, exist_ok=True)

    df_fam_surv = df_surv.copy()
    for f in patient_family_scores.columns:
        df_fam_surv[f"{f}_group"] = (df_fam_surv[f] >= df_fam_surv[f].median()).map({True: "High", False: "Low"})

    def plot_km_family(df, family):
        kmf = KaplanMeierFitter()
        plt.figure(figsize=(5, 4))
        for grp in ["High", "Low"]:
            mask = df[f"{family}_group"] == grp
            kmf.fit(df.loc[mask, "time"], df.loc[mask, "event"], label=f"{family} {grp}")
            kmf.plot_survival_function(ci_show=False)
        g1 = df[f"{family}_group"] == "High"
        g2 = df[f"{family}_group"] == "Low"
        res = logrank_test(df.loc[g1, "time"], df.loc[g2, "time"], df.loc[g1, "event"], df.loc[g2, "event"])
        plt.title(f"{family}\nLog-rank p = {res.p_value:.2e}")
        plt.xlabel("Days")
        plt.ylabel("Overall Survival")
        plt.tight_layout()
        plt.savefig(os.path.join(km_dir, f"KM_{family}.pdf"))
        plt.close()

    for fam in patient_family_scores.columns:
        plot_km_family(df_fam_surv, fam)


def gene_pathway_heatmaps(saliency_pathway_matrix, gene_names, pathway_names, patient_cluster_scores, row_labels, col_labels, output_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import os

    df_heat = pd.DataFrame(saliency_pathway_matrix, index=pathway_names, columns=gene_names)
    os.makedirs(output_path, exist_ok=True)

    # Dendrogram + heatmap
    sns.clustermap(df_heat, cmap="Reds", metric="cosine", method="average", figsize=(10, 8), linewidths=0.1)
    plt.savefig(os.path.join(output_path, "gene_pathway_dendrogram_heatmap.pdf"))
    plt.close()

    sns.clustermap(
        df_heat,
        cmap="vlag",
        center=0,
        metric="cosine",
        method="average",
        z_score=1,
        figsize=(11, 9),
        dendrogram_ratio=(0.15, 0.15),
        cbar_kws={"label": "Saliency × Enrichment"}
    )
    plt.savefig(os.path.join(output_path, "gene_pathway_dendrogram_heatmap.png"), dpi=300)
    plt.close()

    sns.clustermap(
        patient_cluster_scores,
        cmap="vlag",
        metric="correlation",
        z_score=1,
        figsize=(8, 10),
        col_cluster=False
    )
    plt.savefig(os.path.join(output_path, "patient_cluster_dendrogram.pdf"))
    plt.close()

    df_reordered = df_heat.iloc[np.argsort(row_labels), np.argsort(col_labels)]
    sns.heatmap(df_reordered, cmap="Reds", yticklabels=True, xticklabels=True)
    plt.title("Spectral Biclustering Heatmap")
    plt.savefig(os.path.join(output_path, "bicluster_heatmap.pdf"))
    plt.close()


def plot_gene_pathway_modules(df_enrich, output_path):
    os.makedirs(output_path, exist_ok=True)
    build_and_plot_gene_pathway_modules(
        df_enrich=df_enrich,
        output_path=os.path.join(output_path, "gene_pathway_modules"),
        top_genes=30,
        top_pathways=25
    )

    build_and_plot_gene_pathway_umap(
        df_enrich=df_enrich,
        output_path=os.path.join(output_path, "gene_pathway_umap"),
        top_genes=40,
        top_pathways=40
    )

    # plot_pathway_enrichment_dotplot(df_enrich=df_enrich, output_path=output_path, top_k=10)


def cox_pathway_family(patient_family_scores, df_family_surv, output_path):
    import os
    import numpy as np
    import pandas as pd
    from lifelines import CoxPHFitter
    from statsmodels.stats.multitest import multipletests
    import matplotlib.pyplot as plt

    os.makedirs(output_path, exist_ok=True)

    cox_results = []
    for fam in patient_family_scores.columns:
        df_tmp = df_family_surv[["time", "event", fam]].dropna()
        if df_tmp[fam].std() == 0:
            continue
        cph = CoxPHFitter()
        cph.fit(df_tmp, duration_col="time", event_col="event")
        s = cph.summary.loc[fam]
        cox_results.append({
            "Family": fam,
            "HR": s["exp(coef)"],
            "CI_lower": s["exp(coef) lower 95%"],
            "CI_upper": s["exp(coef) upper 95%"],
            "p": s["p"]
        })

    df_cox_family = pd.DataFrame(cox_results)
    df_cox_family["FDR"] = multipletests(df_cox_family["p"], method="fdr_bh")[1]
    df_cox_family = df_cox_family.sort_values("p")
    df_cox_family.to_csv(os.path.join(output_path, "cox_pathway_families.csv"), index=False)

    df_plot = df_cox_family.copy()
    plt.figure(figsize=(6, 0.5 * len(df_plot)))
    y = np.arange(len(df_plot))
    plt.errorbar(
        df_plot["HR"],
        y,
        xerr=[df_plot["HR"] - df_plot["CI_lower"], df_plot["CI_upper"] - df_plot["HR"]],
        fmt="o",
        color="black",
        ecolor="black",
        capsize=3
    )
    plt.axvline(1, linestyle="--", color="red")
    plt.yticks(y, df_plot["Family"])
    plt.xlabel("Hazard Ratio")
    plt.title("Pathway-Family Cox Regression")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cox_pathway_family_forest.pdf"))
    plt.close()


def map_genes_to_clusters(saliency_pathway_matrix, bicluster, top_k=20):
    from collections import defaultdict
    import numpy as np

    gene_cluster_map = defaultdict(list)

    for p_idx, cluster_id in enumerate(bicluster.row_labels_):
        gene_scores = saliency_pathway_matrix[p_idx]
        top_genes = np.argsort(gene_scores)[::-1][:top_k]
        gene_cluster_map[cluster_id].extend(top_genes)

    # Deduplicate
    for c in gene_cluster_map:
        gene_cluster_map[c] = list(set(gene_cluster_map[c]))

    return gene_cluster_map


def plot_cluster_sankey(df_cluster, cluster_name, output_path, cluster_colors=None, pathway_family_map=None):

    os.makedirs(output_path, exist_ok=True)

    if pathway_family_map is None:
        pathway_family_map = {}

    # Map pathways to families
    df_cluster["PathwayFamily"] = df_cluster["Pathway"].apply(lambda x: pathway_family_map.get(x, "Other"))

    families = sorted(df_cluster["PathwayFamily"].unique())
    pathways = sorted(df_cluster["Pathway"].unique())
    genes = sorted(df_cluster["Gene"].unique())
    labels = [cluster_name] + families + pathways + genes
    label_to_id = {l: i for i, l in enumerate(labels)}

    source, target, value = [], [], []

    # Cluster → Family
    cf = df_cluster.groupby("PathwayFamily")["Value"].sum().reset_index()
    for _, r in cf.iterrows():
        source.append(label_to_id[cluster_name])
        target.append(label_to_id[r["PathwayFamily"]])
        value.append(r["Value"])

    # Family → Pathway
    fp = df_cluster.groupby(["PathwayFamily", "Pathway"])["Value"].sum().reset_index()
    for _, r in fp.iterrows():
        source.append(label_to_id[r["PathwayFamily"]])
        target.append(label_to_id[r["Pathway"]])
        value.append(r["Value"])

    # Pathway → Gene
    pg = df_cluster.groupby(["Pathway", "Gene"])["Value"].sum().reset_index()
    for _, r in pg.iterrows():
        source.append(label_to_id[r["Pathway"]])
        target.append(label_to_id[r["Gene"]])
        value.append(r["Value"])

    # Node colors
    if cluster_colors is None:
        cluster_colors = {cluster_name: "#0077B6"}
    node_colors = [
        cluster_colors.get(lbl, "#ADB5BD" if lbl in families else "#CED4DA" if lbl in pathways else "#E9ECEF")
        for lbl in labels
    ]

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(label=labels, color=node_colors, pad=15, thickness=14, line=dict(color="black", width=0.3)),
            link=dict(source=source, target=target, value=value)
        )
    )

    fig.update_layout(title=f"{cluster_name}: Pathway–Gene Attribution", font=dict(size=11, family="Times New Roman"), width=1100, height=750)
    fig.write_image(os.path.join(output_path, f"sankey_{cluster_name}.svg"), scale=2)
    fig.write_image(os.path.join(output_path, f"sankey_{cluster_name}.pdf"), scale=2)


from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_km_clusters(df_surv, patient_cluster_scores, output_path):
    """
    Plots Kaplan–Meier curves for each cluster based on median stratification,
    saves SVG and PDF, and outputs log-rank p-values.

    Parameters
    ----------
    df_surv : pd.DataFrame
        Survival DataFrame with columns ["time", "event"] indexed by patient.
    patient_cluster_scores : pd.DataFrame
        Patient × cluster score matrix (values to stratify patients).
    output_path : str
        Directory to save KM plots and p-values CSV.
    """
    km_dir = os.path.join(output_path, "survival")
    os.makedirs(km_dir, exist_ok=True)

    # Store log-rank p-values
    results = []

    for cluster_id in patient_cluster_scores.columns:
        df = df_surv.copy()
        median_score = df[cluster_id].median()
        df["group"] = df[cluster_id] >= median_score

        # KM curves
        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()

        plt.figure(figsize=(5, 4))
        kmf_high.fit(
            df[df["group"]]["time"],
            df[df["group"]]["event"],
            label="High activity"
        )
        kmf_low.fit(
            df[~df["group"]]["time"],
            df[~df["group"]]["event"],
            label="Low activity"
        )
        kmf_high.plot(ci_show=False)
        kmf_low.plot(ci_show=False)

        # Log-rank test
        result = logrank_test(
            df[df["group"]]["time"],
            df[~df["group"]]["time"],
            df[df["group"]]["event"],
            df[~df["group"]]["event"]
        )

        # Title with p-value
        plt.title(f"Cluster {cluster_id} Survival\nlog-rank p = {result.p_value:.3e}")
        plt.xlabel("Time (days)")
        plt.ylabel("Survival probability")
        plt.tight_layout()

        # Save plots
        for fmt in ["svg", "pdf"]:
            plt.savefig(os.path.join(km_dir, f"cluster_{cluster_id}_survival.{fmt}"))
        plt.close()

        # Save log-rank p-value
        results.append([cluster_id, result.p_value])

    # Save all p-values
    pd.DataFrame(results, columns=["Cluster", "LogRank_p"]).to_csv(
        os.path.join(km_dir, "cluster_survival_pvalues.csv"),
        index=False
    )
    print(f"✅ KM plots and log-rank p-values saved in {km_dir}")


def plot_joint_gene_pathway_umap(
    enrich_matrix,
    pathway_saliency,
    gene_names,
    pathway_names,
    output_path
):
    import umap
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    n_genes, n_pathways = enrich_matrix.shape

    if len(pathway_saliency) != n_pathways:
        raise ValueError("pathway_saliency length must match number of pathways")

    gene_features = enrich_matrix

    pathway_features = np.zeros((n_pathways, n_pathways))
    np.fill_diagonal(pathway_features, pathway_saliency)

    X = np.vstack([gene_features, pathway_features])

    labels = (
        ["Gene"] * n_genes +
        ["Pathway"] * n_pathways
    )

    names = gene_names + pathway_names

    umap_model = umap.UMAP(
        n_neighbors=20,
        min_dist=0.2,
        n_components=2,
        random_state=42
    )

    X_umap = umap_model.fit_transform(X)

    plt.figure(figsize=(6, 5))

    gene_mask = np.array(labels) == "Gene"
    path_mask = np.array(labels) == "Pathway"

    plt.scatter(
        X_umap[gene_mask, 0],
        X_umap[gene_mask, 1],
        s=12,
        alpha=0.6,
        label="Genes"
    )

    plt.scatter(
        X_umap[path_mask, 0],
        X_umap[path_mask, 1],
        s=60,
        marker="^",
        alpha=0.9,
        label="Pathways"
    )

    plt.legend(frameon=False)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("Joint Gene–Pathway UMAP")

    
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "UMAP_gene_pathway_joint.pdf")
    )
    plt.close()
