import argparse
from src.data_loader import load_graph_data
# from src.train_relevance_score import train_and_evaluate
from src.train import train_and_evaluate
# from src.train_heatmap_almost_pass import train_and_evaluate
# from src.train_clean import train_and_evaluate
# from src.train_heatmap_log_ticks_clean import train_and_evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MLP Predictor')
    parser.add_argument('--input-size', type=int, default=128, help='Input size for the first linear layer')
    parser.add_argument('--hidden-size', type=int, default=16, help='Hidden size for the first linear layer')
    parser.add_argument('--dim-latent', type=int, default=128, help='Dimensionality of the latent space in GCNModel')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of layers in GCNModel')
    parser.add_argument('--do-train', type=bool, default=True, help='Training mode for GCNModel')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for the optimizer')
    args = parser.parse_args()

    ##G_dgl, node_features = load_graph_data('./data/neo4j_graph.json')
    ##G_dgl, node_features = load_graph_data('gat/data/pathway_kg_.json')
    G_dgl, node_features = load_graph_data('../gat/data/neo4j_graph_pass.json')
    # G_dgl, node_features = load_graph_data('../gat/data/neo4j_triplets_head1_dim128_lay2_epo20.json')
    

    # Display graph information
    print(f'NumNodes: {G_dgl.num_nodes()}')
    print(f'NumEdges: {G_dgl.num_edges()}')
    print(f'NumFeats: {node_features.size(1)}')

    train_and_evaluate(args, G_dgl, node_features)

                                              
# (kg39) ericsali@erics-MacBook-Pro-4 link_prediction_gcn_pathway % python main.py --num-layers 6 --lr 0.001 --input-size 2 --hidden-size 16 --epochs 10
# NumNodes: 18208
# NumEdges: 19169
# NumFeats: 128
# Traceback (most recent call last):
#   File "/Users/ericsali/Documents/2024_Winter/Project_gnn/reactome_markers/gnn_pathways/link_prediction_gcn_pathway/main.py", line 31, in <module>
#     train_and_evaluate(args, G_dgl, node_features)
#   File "/Users/ericsali/Documents/2024_Winter/Project_gnn/reactome_markers/gnn_pathways/link_prediction_gcn_pathway/src/train.py", line 206, in train_and_evaluate
#     adj_neg = 1 - adj.todense() - np.eye(G_dgl.number_of_nodes())
# ValueError: operands could not be broadcast together with shapes (18191,18208) (18208,18208) 

## python link_prediction_gcn/main.py --lr 0.0001 --epochs 10000