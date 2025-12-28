import argparse
from src.data_loader import load_graph_data
from src.train import train_and_evaluate

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description='MLP Predictor')
    parser.add_argument('--in-feats', type=int, default=128, help='Dimension of the first layer')
    parser.add_argument('--out-feats', type=int, default=128, help='Dimension of the final layer')
    parser.add_argument('--num-heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of layers in the model')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--input-size', type=int, default=2, help='Input size for the first linear layer')
    parser.add_argument('--hidden-size', type=int, default=16, help='Hidden size for the hidden linear layer')
    parser.add_argument('--feat-drop', type=float, default=0.0, help='Feature dropout rate')
    parser.add_argument('--attn-drop', type=float, default=0.0, help='Attention dropout rate')

    args = parser.parse_args()

    # Load graph data (switch between raw graph or precomputed embeddings)
    # Example 1: load from graph JSON
    # G_dgl, node_features = load_graph_data('data/neo4j_graph_pass.json')

    # G_dgl, node_features = load_graph_data(
    #     'embedding/results/node_embeddings/neo4j_triplets_head1_dim128_lay2_epo20.json'
    #     # '../data/neo4j_triplets_head1_dim128_lay2_epo20.json'
    # )
    G_dgl, node_features = load_graph_data('../gat/data/neo4j_graph_pass.json')

    # Train and evaluate model
    train_and_evaluate(args, G_dgl, node_features)
