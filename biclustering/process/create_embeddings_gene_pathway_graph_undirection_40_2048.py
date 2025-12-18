import csv
import json
import os
import numpy as np

# File paths
source_csv_path = 'data/gene_pathway_network_embeddings/embeddings_concat_dim2048_common.csv'
target_csv_path = 'data/gene_pathway_network_embeddings/embeddings_concat_dim2048_common.csv'
relation_csv_path = 'data/gene_pathway_network_embeddings/pathway_gene_type_drivers_filled_to40_tp53.csv'
output_json_path = 'data/multiomics_meth/gene_pathway_graph_random_40_2048_tp53.json'

# Interaction types and corresponding labels
interaction_labels = {
    "0": 0,
    "1": 1,
    "2": -1,
    "3": -1
}

# ---- Safe embedding reader ----
def read_embeddings(file_path):
    embeddings = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Skip the header
        for row in reader:
            name = row[0]
            # Convert to floats and replace NaN/Inf with 0
            embedding = np.array(row[1:], dtype=float)
            if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
            embeddings[name] = embedding.tolist()
    return embeddings

# Read source and target embeddings
source_embeddings = read_embeddings(source_csv_path)
target_embeddings = read_embeddings(target_csv_path)

# Read relationships and collect nodes/edges
nodes = set()
edges = set()
relationships_to_include = []

with open(relation_csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        source_stId = row['Gene1']
        target_stId = row['Gene2']
        relation_type = row['gene_type']

        if relation_type in interaction_labels:
            nodes.update([source_stId, target_stId])
            edge = tuple(sorted([source_stId, target_stId]))
            if edge not in edges:
                edges.add(edge)
                relationships_to_include.append((source_stId, target_stId, relation_type))
                relationships_to_include.append((target_stId, source_stId, relation_type))

print(f"Number of nodes: {len(nodes)}")
print(f"Number of edges: {len(edges)}")

# Build JSON structure safely
relationships = []
for source_stId, target_stId, relation_type in relationships_to_include:
    if source_stId in source_embeddings and target_stId in target_embeddings:
        source_label = interaction_labels[relation_type]
        target_label = 1 if relation_type == "2" else 0 if relation_type == "0" else None

        relationship = {
            "source": {
                "properties": {
                    "name": source_stId,
                    "label": source_label,
                    "embedding": source_embeddings[source_stId]
                }
            },
            "relation": {"type": relation_type},
            "target": {
                "properties": {
                    "name": target_stId,
                    "label": target_label,
                    "embedding": target_embeddings[target_stId]
                }
            }
        }
        relationships.append(relationship)

# Save JSON
with open(output_json_path, 'w') as json_file:
    json.dump(relationships, json_file, indent=2)

print(f"✅ JSON file saved to {output_json_path}")
