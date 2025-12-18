# =========================
# Standard library
# =========================
import os
import json
import itertools
from collections import defaultdict

from itertools import combinations
from tqdm import tqdm
# =========================
# Numerical / scientific
# =========================
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import sem

# =========================
# Deep learning / graphs
# =========================
import torch
import dgl
import networkx as nx

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


def train_and_evaluate(args, G_dgl, node_features):

    import os
    import itertools
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import torch
    import dgl

    from sklearn.cluster import SpectralBiclustering
    from collections import defaultdict

    ########################################
    # 1. EDGE-LEVEL TRAIN / VAL / TEST SPLIT
    ########################################

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

    model = GINNetModel(
        graph=G_dgl,
        in_feats=node_features.shape[1],
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

    relevance_scores = compute_relevance_scores(
        model=model,
        graph=G_dgl,
        features=G_dgl.ndata["feat"],
        node_indices=None,
        method="saliency",     # can switch to "integrated_gradients"
        use_abs=True,
        baseline=None,
        steps=50
    )

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

    edge_ig = edge_integrated_gradients(
        h_test,
        u_all,
        v_all,
        pred,
        steps=50
    ).detach().cpu().numpy()

    edge_ig_norm = (edge_ig - edge_ig.min()) / (edge_ig.max() - edge_ig.min() + 1e-9)

    edge_list = list(zip(
        u_all.cpu().numpy(),
        v_all.cpu().numpy(),
        edge_ig_norm
    ))

    edge_list.sort(key=lambda x: x[2], reverse=True)

    pd.DataFrame(
        edge_list[:50],
        columns=["Gene_u", "Gene_v", "IG_Score"]
    ).to_csv(
        os.path.join(output_path, "top_edge_attributions.csv"),
        index=False
    )

    ########################################
    # 6. PATHWAY ENRICHMENT MATRIX
    ########################################

    df_enrich = pd.read_csv(
        "data/gene_gene_pairs_with_pathwayA_enrichment_unique_20_limit.csv"
    )

    df_enrich = df_enrich[df_enrich["significance"] == "significant"]
    df_enrich["enrich_score"] = -np.log10(df_enrich["pvalue"])

    gene_names = sorted(df_enrich["Gene2"].unique())
    pathway_names = sorted(df_enrich["PathwayB"].unique())

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    pathway_to_idx = {p: i for i, p in enumerate(pathway_names)}

    enrich_matrix = np.zeros((len(gene_names), len(pathway_names)))

    for _, row in df_enrich.iterrows():
        enrich_matrix[
            gene_to_idx[row["Gene2"]],
            pathway_to_idx[row["PathwayB"]]
        ] = row["enrich_score"]

    ########################################
    # 7. GENE × PATHWAY SALIENCY MATRIX
    ########################################

    saliency_pathway_matrix = np.zeros(
        (len(pathway_names), len(gene_names))
    )

    for g, gi in gene_to_idx.items():
        for p, pi in pathway_to_idx.items():
            saliency_pathway_matrix[pi, gi] = (
                saliency_np[gi] * enrich_matrix[gi, pi]
            )

    ########################################
    # 8. BICLUSTERING ON EXPLANATIONS
    ########################################

    bicluster = SpectralBiclustering(
        n_clusters=(10, 10),
        method="log",
        random_state=0
    )

    bicluster.fit(saliency_pathway_matrix)

    row_labels = bicluster.row_labels_
    col_labels = bicluster.column_labels_

    np.save(
        os.path.join(output_path, "bicluster_row_labels.npy"),
        row_labels
    )

    ########################################
    # 9. TOP GENE–PATHWAY PAIRS
    ########################################

    rows = []
    for p_idx in range(len(pathway_names)):
        top_genes = np.argsort(
            saliency_pathway_matrix[p_idx]
        )[::-1][:50]

        for g_idx in top_genes:
            rows.append([
                pathway_names[p_idx],
                gene_names[g_idx],
                saliency_pathway_matrix[p_idx, g_idx]
            ])

    pd.DataFrame(
        rows,
        columns=["Pathway", "Gene", "Saliency"]
    ).to_csv(
        os.path.join(output_path, "top_gene_pathway_pairs.csv"),
        index=False
    )

    ########################################
    # 10. CLUSTER → PATHWAY → GENE FLOWS
    ########################################

    flows = []

    for p_idx, c_id in enumerate(row_labels):
        scores = saliency_pathway_matrix[p_idx]
        top_genes = np.argsort(scores)[::-1][:50]

        for g_idx in top_genes:
            if scores[g_idx] > 0:
                flows.append([
                    f"Cluster {c_id}",
                    pathway_names[p_idx],
                    gene_names[g_idx],
                    scores[g_idx]
                ])

    pd.DataFrame(
        flows,
        columns=["Cluster", "Pathway", "Gene", "Value"]
    ).to_csv(
        os.path.join(output_path, "sankey_cluster_pathway_gene.csv"),
        index=False
    )

    print("✅ Training, attribution, enrichment, and biclustering complete.")


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

    bicluster = SpectralBiclustering(
        n_clusters=(10, 10),
        method="log",
        random_state=0
    )

    bicluster.fit(saliency_pathway_matrix)

    # Row = pathways, Column = genes
    row_labels = bicluster.row_labels_
    col_labels = bicluster.column_labels_


    # Gene names from enrichment matrix
    gene_names = sorted(df_enrich["Gene2"].unique())

    # Assign gene → cluster from biclustering
    gene_clusters = pd.Series(
        col_labels,
        index=gene_names
    )

    # Optional sanity check
    print(gene_clusters.value_counts())

    # Order indices by cluster
    row_order = np.argsort(row_labels)
    col_order = np.argsort(col_labels)

    saliency_reordered = saliency_pathway_matrix[
        row_order, :
    ][:, col_order]

    pathways_reordered = [pathway_names[i] for i in row_order]
    genes_reordered = [gene_names[i] for i in col_order]


    bicluster_gene_ids = gene_names
    bicluster_pathway_ids = pathway_names

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

    rows = []
    for p in range(saliency_pathway_matrix.shape[0]):
        top_genes = np.argsort(
            saliency_pathway_matrix[p]
        )[::-1][:50]

        for g in top_genes:
            rows.append([
                pathway_names[p],
                gene_names[g],
                saliency_pathway_matrix[p, g]
            ])

    df_gp = pd.DataFrame(
        rows,
        columns=["Pathway", "Gene", "Saliency"]
    )

    df_gp.to_csv(
        os.path.join(output_path, "top_gene_pathway_pairs.csv"),
        index=False
    )

    top_k_genes = 50
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

    gene_cluster_map = defaultdict(list)

    for p_idx, cluster_id in enumerate(bicluster.row_labels_):
        gene_scores = saliency_pathway_matrix[p_idx]
        top_genes = np.argsort(gene_scores)[::-1][:50]

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
        top_genes = np.argsort(gene_scores)[::-1][:50]

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
    
    patient_cluster_scores = {}

    for cluster_id, gene_idxs in gene_cluster_map.items():
        if len(gene_idxs) == 0:
            continue

        patient_cluster_scores[f"Cluster_{cluster_id}"] = (
            expr_matrix[:, gene_idxs].mean(axis=1)
        )
        
    patient_cluster_scores = pd.DataFrame(
        patient_cluster_scores,
        index=patient_ids
    )

    assert patient_cluster_scores.shape[0] == len(patient_ids)

    df_surv = surv.join(patient_cluster_scores, how="inner")

    print(f"Survival patients total: {surv.shape[0]}")
    print(f"Bicluster patients: {patient_cluster_scores.shape[0]}")
    print(f"After alignment: {df_surv.shape[0]}")
    
    y_struct = Surv.from_arrays(
        event=df_surv["event"].astype(bool).values,
        time=df_surv["time"].values
    )

        

    y_train = y_struct
    y_test = y_struct

    for col in patient_cluster_scores.columns:
        risk_scores = df_surv[col].values

        cindex_ipcw = concordance_index_ipcw(
            y_train,
            y_test,
            -risk_scores
        )[0]

        cindex_harrell = concordance_index(
            df_surv["time"],
            -risk_scores,
            df_surv["event"]
        )

        times = np.linspace(
            df_surv["time"].quantile(0.05),
            df_surv["time"].quantile(0.95),
            10
        )

        auc, mean_auc = cumulative_dynamic_auc(
            y_train,
            y_test,
            -risk_scores,
            times
        )

        print(
            f"{col} | IPCW C-index: {cindex_ipcw:.3f}, "
            f"Harrell C-index: {cindex_harrell:.3f}, "
            f"Mean AUC: {mean_auc:.3f}"
        )

        plt.figure(figsize=(5, 4))
        plt.plot(times / 365.0, auc, marker="o")
        plt.xlabel("Years")
        plt.ylabel("Time-dependent AUC")
        plt.title(f"{col} Survival ROC")
        plt.ylim(0.4, 1.0)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_path, f"ROC_time_{col}.pdf")
        )
        plt.close()


    ########################################
    # SAFE INTERSECTION
    ########################################

    common_patients = (
        set(patient_cluster_scores.index)
        & set(surv.index)
    )


    common_patients = sorted(common_patients)

    ########################################
    # ORDER PATIENTS BY BICLUSTER
    ########################################

    ordered_patients = (
        patient_cluster_scores
        .mean(axis=1)
        .sort_values(ascending=False)
        .index
    )


    heatmap_df = patient_cluster_scores.loc[ordered_patients]

    heatmap_df = (heatmap_df - heatmap_df.mean()) / heatmap_df.std()

    surv_event = surv.loc[ordered_patients, "event"]

    # Color map for survival status
    surv_colors = {
        0: "#4daf4a",  # censored
        1: "#e41a1c"   # event
    }

    row_colors = surv_event.map(surv_colors)

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        heatmap_df,
        cmap="vlag",
        center=0,
        xticklabels=True,
        yticklabels=False,
        cbar_kws={"label": "Z-scored Expression"}
    )

    plt.xlabel("Gene Clusters")
    plt.ylabel("Patients (ordered by bicluster)")
    plt.title("Patient × Cluster Expression Heatmap (Bicluster-aligned)")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_path,
            "heatmap_patient_cluster_bicluster_aligned.pdf"
        )
    )
    plt.close()

    ########################################
    # 17. HIGH / LOW STRATIFICATION
    ########################################

    cluster_cols = patient_cluster_scores.columns

    for c in cluster_cols:
        df_surv[f"{c}_group"] = (
            df_surv[c] >= df_surv[c].median()
        ).map({True: "High", False: "Low"})

    ########################################
    # 18. KAPLAN–MEIER SURVIVAL ANALYSIS
    ########################################


    km_dir = os.path.join(output_path, "KM_plots")
    os.makedirs(km_dir, exist_ok=True)

    for c in cluster_cols:
        plot_km(df_surv, c, output_path)
            

    # patient → bicluster assignment
    patient_bicluster = (
        patient_cluster_scores
        .idxmax(axis=1)
        .loc[common_patients]
    )
        
            
    plot_patient_bicluster_heatmap(
        patient_cluster_scores=patient_cluster_scores,
        patient_bicluster=patient_bicluster,
        output_path=output_path
    )


    # build_and_plot_gene_pathway_modules_centered(
    #     df_enrich,
    #     os.path.join(output_path, "pathway_centered_gene_modules")
    # )

    build_and_plot_gene_pathway_modules_centered(
        df_enrich=df_enrich,
        output_path=os.path.join(
            output_path, "pathway_centered_gene_modules"
        ),
        top_genes_per_pathway=50,
        top_pathways=4
    )


    umap_dir = os.path.join(output_path, "UMAP")
    os.makedirs(umap_dir, exist_ok=True)

    X = patient_cluster_scores.loc[df_surv.index].values

    umap_model = umap.UMAP(
        n_neighbors=15,
        min_dist=0.2,
        n_components=2,
        random_state=42
    )

    X_umap = umap_model.fit_transform(X)

    df_umap = pd.DataFrame(
        X_umap,
        index=df_surv.index,
        columns=["UMAP1", "UMAP2"]
    )

    df_umap["time"] = df_surv["time"]
    df_umap["event"] = df_surv["event"]


    plt.figure(figsize=(5, 4))

    for ev, label in [(0, "Alive / Censored"), (1, "Dead")]:
        mask = df_umap["event"] == ev
        plt.scatter(
            df_umap.loc[mask, "UMAP1"],
            df_umap.loc[mask, "UMAP2"],
            label=label,
            s=25,
            alpha=0.8
        )

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend(frameon=False)
    plt.title("UMAP of Patients (Event Status)")

    plt.tight_layout()
    plt.savefig(
        os.path.join(umap_dir, "UMAP_bicluster_event.pdf")
    )
    plt.close()



    plt.figure(figsize=(5, 4))
    sc = plt.scatter(
        df_umap["UMAP1"],
        df_umap["UMAP2"],
        c=df_umap["time"] / 365.0,
        cmap="viridis",
        s=25,
        alpha=0.8
    )

    plt.colorbar(sc, label="OS time (years)")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP of Patients (Bicluster Scores)")

    plt.tight_layout()
    plt.savefig(
        os.path.join(umap_dir, "UMAP_bicluster_OS_time.pdf")
    )
    plt.close()



    dca_dir = os.path.join(output_path, "DCA")
    os.makedirs(dca_dir, exist_ok=True)

    thresholds = np.linspace(0.01, 0.99, 100)

    event = df_surv["event"].values.astype(int)

    # Reference strategies
    treat_all = event.mean() - (1 - event.mean()) * (thresholds / (1 - thresholds))
    treat_none = np.zeros_like(thresholds)

    for col in patient_cluster_scores.columns:
        risk = df_surv[col].values

        nb_model = decision_curve_analysis(event, risk, thresholds)

        plt.figure(figsize=(5, 4))

        plt.plot(thresholds, nb_model, label=col, linewidth=2)
        plt.plot(thresholds, treat_all, "--", label="Treat All", alpha=0.7)
        plt.plot(thresholds, treat_none, "--", label="Treat None", alpha=0.7)

        plt.xlabel("Threshold Probability")
        plt.ylabel("Net Benefit")
        plt.title(f"Decision Curve Analysis – {col}")
        plt.legend(frameon=False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(dca_dir, f"DCA_{col}.pdf")
        )
        plt.close()


    risk_dir = os.path.join(output_path, "risk_distributions")
    os.makedirs(risk_dir, exist_ok=True)

    # Map event to readable labels
    df_surv["event_label"] = df_surv["event"].map({
        0: "Alive / Censored",
        1: "Dead"
    })

    for col in patient_cluster_scores.columns:
        plt.figure(figsize=(4.5, 4))

        sns.violinplot(
            data=df_surv,
            x="event_label",
            y=col,
            inner=None,
            cut=0
        )

        sns.boxplot(
            data=df_surv,
            x="event_label",
            y=col,
            width=0.2,
            showcaps=True,
            boxprops={"facecolor": "none"},
            showfliers=False
        )

        plt.xlabel("")
        plt.ylabel("Bicluster Risk Score")
        plt.title(f"{col} Risk Score Distribution")

        plt.tight_layout()
        plt.savefig(
            os.path.join(risk_dir, f"risk_violin_{col}.pdf")
        )
        plt.close()


    na_dir = os.path.join(output_path, "NA_plots")
    os.makedirs(na_dir, exist_ok=True)

    def plot_nelson_aalen(df, cluster):
        naf = NelsonAalenFitter()
        plt.figure(figsize=(5, 4))

        for grp in ["High", "Low"]:
            mask = df[f"{cluster}_group"] == grp
            naf.fit(
                durations=df.loc[mask, "time"],
                event_observed=df.loc[mask, "event"],
                label=f"{cluster} {grp}"
            )
            naf.plot_cumulative_hazard(ci_show=False)

        plt.xlabel("Days")
        plt.ylabel("Cumulative Hazard")
        plt.title(f"{cluster} – Nelson–Aalen")

        plt.tight_layout()
        plt.savefig(
            os.path.join(na_dir, f"NA_{cluster}.pdf")
        )
        plt.close()

    for c in cluster_cols:
        plot_nelson_aalen(df_surv, c)
        
    ########################################
    # 19. COX PROPORTIONAL HAZARDS (UNIVARIATE)
    ########################################

    cox_results = []

    for c in cluster_cols:
        df_tmp = df_surv[["time", "event", c]].dropna()
        cph = CoxPHFitter()
        cph.fit(
            df_tmp,
            duration_col="time",
            event_col="event"
        )

        s = cph.summary.loc[c]
        cox_results.append({
            "Cluster": c,
            "HR": s["exp(coef)"],
            "CI_lower": s["exp(coef) lower 95%"],
            "CI_upper": s["exp(coef) upper 95%"],
            "p": s["p"]
        })

    df_cox = pd.DataFrame(cox_results).sort_values("p")
    df_cox.to_csv(
        os.path.join(output_path, "cox_univariate_clusters.csv"),
        index=False
    )

    print("✅ Biclustering → Sankey → Survival pipeline completed.")



    ########################################
    # 20. PATHWAY → FAMILY MAP
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

    def map_to_family(pathway):
        return PATHWAY_FAMILY_MAP.get(pathway, "Other")

    ########################################
    # 21. PATIENT × PATHWAY SCORES (FIXED)
    ########################################

    patient_pathway_scores = {}

    for p_idx, pathway in enumerate(pathway_names):
        gene_scores = saliency_pathway_matrix[p_idx]
        top_gene_idxs = np.argsort(gene_scores)[::-1][:50]

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

    ########################################
    # 26. FOREST PLOT
    ########################################

    df_plot = df_cox_family.copy()
    df_plot["logHR"] = np.log(df_plot["HR"])

    plt.figure(figsize=(6, 0.5 * len(df_plot)))

    y = np.arange(len(df_plot))

    plt.errorbar(
        df_plot["HR"],
        y,
        xerr=[
            df_plot["HR"] - df_plot["CI_lower"],
            df_plot["CI_upper"] - df_plot["HR"]
        ],
        fmt="o",
        color="black",
        ecolor="black",
        capsize=3
    )

    plt.axvline(1, linestyle="--", color="red")

    plt.yticks(
        y,
        df_plot["Family"]
    )

    plt.xlabel("Hazard Ratio")
    plt.title("Pathway-Family Cox Regression")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "cox_pathway_family_forest.pdf")
    )
    plt.close()

    print("✅ Forest plot saved.")





    module_dir = os.path.join(output_path, "gene_pathway_modules")
    os.makedirs(module_dir, exist_ok=True)

    df_enrich = pd.read_csv(
        "data/gene_gene_pairs_with_pathwayA_enrichment_unique_20_limit.csv"
    )

    build_and_plot_gene_pathway_modules(
        df_enrich=df_enrich,
        output_path=os.path.join(output_path, "gene_pathway_modules"),
        top_genes=30,
        top_pathways=25
    )

    df_enrich = pd.read_csv(
        "data/gene_gene_pairs_with_pathwayA_enrichment_unique_20_limit.csv"
    )

    build_and_plot_gene_pathway_umap(
        df_enrich=df_enrich,
        output_path=os.path.join(output_path, "gene_pathway_umap"),
        top_genes=40,
        top_pathways=40
    )

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # import pandas as pd

    df_heat = pd.DataFrame(
        saliency_pathway_matrix,
        index=pathway_names,
        columns=gene_names
    )

    sns.clustermap(
        df_heat,
        cmap="Reds",
        metric="cosine",
        method="average",
        figsize=(10, 8),
        linewidths=0.1
    )

    plt.savefig("gene_pathway_dendrogram_heatmap.pdf")
    plt.close()



    sns.clustermap(
        df_heat,
        cmap="vlag",
        center=0,
        metric="cosine",
        method="average",
        z_score=1,            # normalize genes
        figsize=(11, 9),
        dendrogram_ratio=(0.15, 0.15),
        cbar_kws={"label": "Saliency × Enrichment"}
    )

    plt.savefig("gene_pathway_dendrogram_heatmap.pdf")
    plt.savefig("gene_pathway_dendrogram_heatmap.png", dpi=300)
    plt.close()

    sns.clustermap(
        patient_cluster_scores,
        cmap="vlag",
        metric="correlation",
        z_score=1,
        figsize=(8, 10),
        col_cluster=False
    )

    plt.savefig("patient_cluster_dendrogram.pdf")
    plt.close()

    row_order = np.argsort(row_labels)
    col_order = np.argsort(col_labels)

    df_reordered = df_heat.iloc[row_order, col_order]

    sns.heatmap(
        df_reordered,
        cmap="Reds",
        yticklabels=True,
        xticklabels=True
    )

    plt.title("Spectral Biclustering Heatmap")
    plt.savefig("bicluster_heatmap.pdf")
    plt.close()



    plot_pathway_enrichment_dotplot(
        df_enrich=df_enrich,
        output_path=output_path,
        top_k=10
    )



    # ########################################
    # # 22. PATIENT × PATHWAY-FAMILY SCORES
    # ########################################

    family_scores = {}

    for pathway, scores in patient_pathway_scores.items():
        fam = map_to_family(pathway)
        if fam not in family_scores:
            family_scores[fam] = []
        family_scores[fam].append(scores)

    patient_family_scores = {
        fam: pd.concat(vals, axis=1).mean(axis=1)
        for fam, vals in family_scores.items()
    }

    patient_family_scores = pd.DataFrame(patient_family_scores)
    patient_family_scores.index = patient_ids

    patient_family_scores.to_csv(
        os.path.join(output_path, "patient_pathway_family_scores.csv")
    )



    ########################################
    # 23. MERGE WITH SURVIVAL
    ########################################

    df_fam_surv = surv.join(
        patient_family_scores,
        how="inner"
    )

    family_cols = patient_family_scores.columns



    ########################################
    # 24. FAMILY STRATIFICATION
    ########################################

    for f in family_cols:
        df_fam_surv[f"{f}_group"] = (
            df_fam_surv[f] >= df_fam_surv[f].median()
        ).map({True: "High", False: "Low"})



    ########################################
    # 25. KM PLOTS – PATHWAY FAMILY
    ########################################

    km_fam_dir = os.path.join(output_path, "KM_family")
    os.makedirs(km_fam_dir, exist_ok=True)

    def plot_km_family(df, family):
        kmf = KaplanMeierFitter()
        plt.figure(figsize=(5, 4))

        for grp in ["High", "Low"]:
            mask = df[f"{family}_group"] == grp
            kmf.fit(
                df.loc[mask, "time"],
                df.loc[mask, "event"],
                label=f"{family} {grp}"
            )
            kmf.plot_survival_function(ci_show=False)

        g1 = df[f"{family}_group"] == "High"
        g2 = df[f"{family}_group"] == "Low"

        res = logrank_test(
            df.loc[g1, "time"],
            df.loc[g2, "time"],
            df.loc[g1, "event"],
            df.loc[g2, "event"]
        )

        plt.title(
            f"{family}\nLog-rank p = {res.p_value:.2e}"
        )
        plt.xlabel("Days")
        plt.ylabel("Overall Survival")

        plt.tight_layout()
        plt.savefig(
            os.path.join(km_fam_dir, f"KM_{family}.pdf")
        )
        plt.close()

    for fam in family_cols:
        plot_km_family(df_fam_surv, fam)


    ########################################
    # 26. COX – PATHWAY FAMILY
    ########################################

    cox_family = []

    for fam in family_cols:
        df_tmp = df_fam_surv[["time", "event", fam]].dropna()
        cph = CoxPHFitter()
        cph.fit(
            df_tmp,
            duration_col="time",
            event_col="event"
        )

        s = cph.summary.loc[fam]
        cox_family.append({
            "PathwayFamily": fam,
            "HR": s["exp(coef)"],
            "CI_lower": s["exp(coef) lower 95%"],
            "CI_upper": s["exp(coef) upper 95%"],
            "p": s["p"]
        })

    df_cox_family = (
        pd.DataFrame(cox_family)
        .sort_values("p")
    )

    df_cox_family.to_csv(
        os.path.join(output_path, "cox_pathway_family.csv"),
        index=False
    )












    ########################################
    # 14. PLOT 3-LEVEL SANKEY
    ########################################


    # Unique labels
    clusters = df_flows["Cluster"].unique().tolist()
    pathways = df_flows["Pathway"].unique().tolist()
    genes = df_flows["Gene"].unique().tolist()

    labels = clusters + pathways + genes

    label_to_id = {l: i for i, l in enumerate(labels)}

    # Build links
    source = []
    target = []
    value = []

    # Cluster → Pathway
    cp = df_flows.groupby(["Cluster", "Pathway"])["Value"].sum().reset_index()
    for _, row in cp.iterrows():
        source.append(label_to_id[row["Cluster"]])
        target.append(label_to_id[row["Pathway"]])
        value.append(row["Value"])

    # Pathway → Gene
    pg = df_flows.groupby(["Pathway", "Gene"])["Value"].sum().reset_index()
    for _, row in pg.iterrows():
        source.append(label_to_id[row["Pathway"]])
        target.append(label_to_id[row["Gene"]])
        value.append(row["Value"])

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=labels,
                    pad=15,
                    thickness=15
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value
                )
            )
        ]
    )

    fig.update_layout(
        title="Cluster → Pathway → Gene Sankey Diagram",
        font_size=10
    )

    fig.write_html(
        os.path.join(output_path, "cluster_pathway_gene_sankey.html")
    )


    # Load Sankey flow table (created earlier)
    df_flows = pd.read_csv(
        os.path.join(output_path, "sankey_cluster_pathway_gene.csv")
    )

    # Unique labels (ordered)
    clusters = sorted(df_flows["Cluster"].unique().tolist())
    pathways = sorted(df_flows["Pathway"].unique().tolist())
    genes = sorted(df_flows["Gene"].unique().tolist())

    labels = clusters + pathways + genes
    label_to_id = {l: i for i, l in enumerate(labels)}


    source = []
    target = []
    value = []

    # Cluster → Pathway
    cp = df_flows.groupby(["Cluster", "Pathway"])["Value"].sum().reset_index()
    for _, r in cp.iterrows():
        source.append(label_to_id[r["Cluster"]])
        target.append(label_to_id[r["Pathway"]])
        value.append(r["Value"])

    # Pathway → Gene
    pg = df_flows.groupby(["Pathway", "Gene"])["Value"].sum().reset_index()
    for _, r in pg.iterrows():
        source.append(label_to_id[r["Pathway"]])
        target.append(label_to_id[r["Gene"]])
        value.append(r["Value"])


    # Example cluster colors (replace with yours if defined globally)
    CLUSTER_COLORS = {
        "Cluster 0": "#0077B6",
        "Cluster 1": "#00B4D8",
        "Cluster 2": "#48CAE4",
        "Cluster 3": "#90DBF4",
    }

    node_colors = []

    for lbl in labels:
        if lbl in CLUSTER_COLORS:
            node_colors.append(CLUSTER_COLORS[lbl])
        elif lbl in pathways:
            node_colors.append("#ADB5BD")   # pathway gray
        else:
            node_colors.append("#DEE2E6")   # gene light gray


    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    label=labels,
                    color=node_colors,
                    pad=15,
                    thickness=14,
                    line=dict(color="black", width=0.3)
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value
                )
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="Cluster → Pathway → Gene Sankey Diagram",
            x=0.5
        ),
        font=dict(
            size=11,
            family="Times New Roman"
        ),
        width=1200,
        height=800,
        margin=dict(l=20, r=20, t=40, b=20)
    )


    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    fig.write_image(
        os.path.join(output_path, "cluster_pathway_gene_sankey.svg"),
        format="svg",
        scale=2
    )

    fig.write_image(
        os.path.join(output_path, "cluster_pathway_gene_sankey.pdf"),
        format="pdf",
        scale=2
    )

    print("✅ Sankey exported as SVG and PDF (publication-ready).")



    ########################################
    # PATHWAY → FAMILY MAPPING
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

    def map_to_family(pathway):
        return PATHWAY_FAMILY_MAP.get(pathway, "Other")


    ########################################
    # LOAD & THRESHOLD FLOWS
    ########################################

    df = pd.read_csv(
        os.path.join(output_path, "sankey_cluster_pathway_gene.csv")
    )

    # Threshold weak flows (global percentile)
    THRESH_PERCENTILE = 75   # keep top 25%
    threshold = np.percentile(df["Value"], THRESH_PERCENTILE)

    df = df[df["Value"] >= threshold].copy()

    # Map pathways to families
    df["PathwayFamily"] = df["Pathway"].apply(map_to_family)


    ########################################
    # CLUSTER-SPECIFIC SANKEY FUNCTION
    ########################################

    def plot_cluster_sankey(
        df_cluster,
        cluster_name,
        output_path
    ):
        # Node sets
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
        fp = df_cluster.groupby(
            ["PathwayFamily", "Pathway"]
        )["Value"].sum().reset_index()
        for _, r in fp.iterrows():
            source.append(label_to_id[r["PathwayFamily"]])
            target.append(label_to_id[r["Pathway"]])
            value.append(r["Value"])

        # Pathway → Gene
        pg = df_cluster.groupby(
            ["Pathway", "Gene"]
        )["Value"].sum().reset_index()
        for _, r in pg.iterrows():
            source.append(label_to_id[r["Pathway"]])
            target.append(label_to_id[r["Gene"]])
            value.append(r["Value"])

        # Colors
        node_colors = []
        for lbl in labels:
            if lbl == cluster_name:
                node_colors.append(CLUSTER_COLORS.get(cluster_name, "#0077B6"))
            elif lbl in families:
                node_colors.append("#ADB5BD")
            elif lbl in pathways:
                node_colors.append("#CED4DA")
            else:
                node_colors.append("#E9ECEF")

        # Plot
        fig = go.Figure(
            go.Sankey(
                arrangement="snap",
                node=dict(
                    label=labels,
                    color=node_colors,
                    pad=15,
                    thickness=14,
                    line=dict(color="black", width=0.3)
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value
                )
            )
        )

        fig.update_layout(
            title=f"{cluster_name}: Pathway–Gene Attribution",
            font=dict(size=11, family="Times New Roman"),
            width=1100,
            height=750
        )

        # Export
        fig.write_image(
            os.path.join(output_path, f"sankey_{cluster_name}.svg"),
            format="svg",
            scale=2
        )

        fig.write_image(
            os.path.join(output_path, f"sankey_{cluster_name}.pdf"),
            format="pdf",
            scale=2
        )



    ########################################
    # GENERATE ALL CLUSTER SANKEYS
    ########################################

    for cluster in sorted(df["Cluster"].unique()):
        df_c = df[df["Cluster"] == cluster]
        plot_cluster_sankey(
            df_cluster=df_c,
            cluster_name=cluster,
            output_path=output_path
        )

    print("✅ Cluster-specific Sankeys exported (SVG + PDF).")
    
    
    ########################################
    # MAP GENES TO CLUSTERS
    ########################################

    

    gene_cluster_map = defaultdict(list)

    # saliency_pathway_matrix: [Pathway × Gene]
    # bicluster.row_labels_: pathway → cluster

    for p_idx, cluster_id in enumerate(bicluster.row_labels_):
        gene_scores = saliency_pathway_matrix[p_idx]
        top_genes = np.argsort(gene_scores)[::-1][:20]  # top genes per pathway

        for g_idx in top_genes:
            gene_cluster_map[cluster_id].append(g_idx)

    # deduplicate
    for c in gene_cluster_map:
        gene_cluster_map[c] = list(set(gene_cluster_map[c]))



    # df_surv = surv.join(patient_cluster_scores, how="inner")


    ########################################
    # PLOT KM CURVES (PER CLUSTER)
    ########################################

    os.makedirs(os.path.join(output_path, "survival"), exist_ok=True)

    for cluster_id in patient_cluster_scores.columns:
        df = df_surv.copy()

        median_score = df[cluster_id].median()
        df["group"] = df[cluster_id] >= median_score

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

        plt.title(
            f"Cluster {cluster_id} Survival\n"
            f"log-rank p = {result.p_value:.3e}"
        )
        plt.xlabel("Time (days)")
        plt.ylabel("Survival probability")

        plt.tight_layout()

        # Export
        plt.savefig(
            os.path.join(
                output_path,
                "survival",
                f"cluster_{cluster_id}_survival.svg"
            )
        )
        plt.savefig(
            os.path.join(
                output_path,
                "survival",
                f"cluster_{cluster_id}_survival.pdf"
            )
        )
        plt.close()

    ########################################
    # SAVE LOG-RANK P-VALUES
    ########################################

    results = []

    for cluster_id in patient_cluster_scores.columns:
        df = df_surv.copy()
        median_score = df[cluster_id].median()
        df["group"] = df[cluster_id] >= median_score

        res = logrank_test(
            df[df["group"]]["time"],
            df[~df["group"]]["time"],
            df[df["group"]]["event"],
            df[~df["group"]]["event"]
        )

        results.append([cluster_id, res.p_value])

    pd.DataFrame(
        results,
        columns=["Cluster", "LogRank_p"]
    ).to_csv(
        os.path.join(output_path, "survival", "cluster_survival_pvalues.csv"),
        index=False
    )










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
        df.sort_values("p_value")
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

def edge_integrated_gradients(
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



