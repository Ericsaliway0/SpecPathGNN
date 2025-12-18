import pandas as pd
from itertools import product

# --- Load enrichment results
enrichment = pd.read_csv("gnn_embedding/results/enrichment/enrichment_simple.csv")
enrichment_dict = enrichment.set_index('stId')[['p_value', 'significance']].to_dict(orient='index')

# --- Load pathway-to-gene mapping
pathway_genes_ = pd.read_csv("data/processed/pathways_mapped.tsv", sep='\t')

# Limit to first 200 non-empty gene columns for each row
limited_rows = []
for _, row in pathway_genes_.iterrows():
    pathway_id = row['PathwayID']
    genes = [g for g in row[1:] if pd.notna(g)]
    genes = genes[:200]
    limited_rows.append([pathway_id] + genes)

# Rebuild DataFrame with fixed columns
max_genes = 20
cols = ['PathwayID'] + [f'Gene{i+1}' for i in range(max_genes)]
limited_df = pd.DataFrame(limited_rows, columns=cols)

# Save back
limited_df.to_csv("data/processed/pathways_mapped_max20.tsv", sep="\t", index=False)
print("✅ Saved limited pathway-gene table with at most 200 genes each")

# --- Reload the trimmed mapping
pathway_genes = pd.read_csv("data/processed/pathways_mapped_max20.tsv", sep='\t')

pathway_to_genes = {}
for idx, row in pathway_genes.iterrows():
    genes = [g for g in row[1:] if pd.notna(g)]
    pathway_to_genes[row['PathwayID']] = genes

# --- Load filtered Reactome pathway relationships
pathway_relations = pd.read_csv("data/processed/ReactomePathwaysRelation_filtered.tsv", sep='\t', header=None)
pathway_relations.columns = ['PathwayA', 'PathwayB']

# --- Keep only relations where PathwayA is in enrichment
pathway_relations = pathway_relations[pathway_relations['PathwayA'].isin(enrichment['stId'])]

# --- Generate gene-gene pairs
gene_pairs = []
for _, row in pathway_relations.iterrows():
    genesA = pathway_to_genes.get(row['PathwayA'], [])
    genesB = pathway_to_genes.get(row['PathwayB'], [])

    # ❗️Skip if either side has no mapped genes
    if not genesA or not genesB:
        continue

    pval = enrichment_dict[row['PathwayA']]['p_value']
    sig = enrichment_dict[row['PathwayA']]['significance']

    for gA, gB in product(genesA, genesB):
        gene_pairs.append([
            row['PathwayA'],
            gA,
            row['PathwayB'],
            gB,
            pval,
            sig
        ])

# --- Build DataFrame
columns = ["PathwayA", "Gene1", "PathwayB", "Gene2", "pvalue", "significance"]
gene_pairs_df = pd.DataFrame(gene_pairs, columns=columns)

# ✅ Remove duplicate gene–gene pairs
gene_pairs_df.drop_duplicates(subset=["PathwayA", "Gene1", "PathwayB", "Gene2"], inplace=True)

# --- Save
output_path = "data/processed/gene_gene_pairs_with_pathwayA_enrichment_unique_20_limit.csv"
gene_pairs_df.to_csv(output_path, index=False, header=True)

print(f"✅ Saved gene-gene pairs with header → {output_path}")
