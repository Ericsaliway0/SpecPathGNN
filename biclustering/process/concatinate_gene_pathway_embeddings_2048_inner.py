import pandas as pd
import numpy as np

# Input paths
file1 = "data/gene_pathway_network_embeddings/embeddings_lr0.001_dim1024_lay2_epo100_final.csv"
file2 = "data/gene_pathway_network_embeddings/embeddings_lr0.0001_dim1024_lay2_epo21_final.csv"

# Output path
output_file = "data/gene_pathway_network_embeddings/embeddings_concat_dim2048_common.csv"

# Load both embeddings
emb1 = pd.read_csv(file1)
emb2 = pd.read_csv(file2)

# Ensure 'Gene' is the column name for merging
emb1.rename(columns={emb1.columns[0]: "Gene"}, inplace=True)
emb2.rename(columns={emb2.columns[0]: "Gene"}, inplace=True)

# Get embedding dimensions
dim1 = emb1.shape[1] - 1
dim2 = emb2.shape[1] - 1

# ✅ Merge only genes that appear in BOTH files
merged = pd.merge(emb1, emb2, on="Gene", how="inner")

# Fill missing numeric values with 0 (optional, for safety)
merged.fillna(0, inplace=True)

# Verify dimension
expected_dim = dim1 + dim2
actual_dim = merged.shape[1] - 1
assert actual_dim == expected_dim, f"Expected {expected_dim}, got {actual_dim}"

# Save result
merged.to_csv(output_file, index=False)

print(f"✅ Concatenated embeddings saved to: {output_file}")
print(f"   Common genes: {merged.shape[0]}, Embedding dimension: {expected_dim}")
