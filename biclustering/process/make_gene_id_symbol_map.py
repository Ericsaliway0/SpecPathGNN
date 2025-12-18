import os
import gzip
import pandas as pd
import urllib.request

# -----------------------------
# 1. Download GENCODE v23 GTF
# -----------------------------
url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_23/gencode.v23.annotation.gtf.gz"
gtf_path = "data/gencode.v23.annotation.gtf.gz"
os.makedirs("data", exist_ok=True)

if not os.path.exists(gtf_path):
    print(f"⬇️  Downloading GENCODE v23 annotation from {url}")
    urllib.request.urlretrieve(url, gtf_path)
    print(f"✅ Downloaded: {gtf_path}")
else:
    print(f"✔️  Found existing {gtf_path}")

# -----------------------------
# 2. Parse GTF → TSV mapping
# -----------------------------
print("🔍 Parsing GTF to extract Ensembl → Gene symbol map...")

records = []
with gzip.open(gtf_path, "rt") as f:
    for line in f:
        if line.startswith("#"):
            continue
        fields = line.strip().split("\t")
        if fields[2] != "gene":
            continue
        attrs = fields[8]
        try:
            gene_id = attrs.split('gene_id "')[1].split('"')[0].split('.')[0]
            gene_name = attrs.split('gene_name "')[1].split('"')[0]
            records.append((gene_id, gene_name))
        except Exception:
            continue

df = pd.DataFrame(records, columns=["gene_id", "gene_symbol"])
df.drop_duplicates(inplace=True)

# -----------------------------
# 3. Save to TSV
# -----------------------------
output_path = "data/gene_id_symbol_map.tsv"
df.to_csv(output_path, sep="\t", index=False)
print(f"✅ Saved mapping to {output_path}")
print(f"Total mappings: {len(df):,}")
print(df.head(10))
