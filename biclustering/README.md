# A Biological Knowledge Graph for Representational Learning 
This repository provides the code for our research project "A Biological Knowledge Graph for Representational Learning".

edge_prediction_project/
│
├── data/
│   └── neo4j_graph.json
│
├── results/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── models.py
│   ├── utils.py
│   └── train.py
│
└── main.py

python auc_sagegraph_bkg/main.py --learning-rate 0.0005 --input-size 2 --hidden-size 128 --epochs 20000

## Data resources
The different dataset and KG used in this project are located in data directory. These files include:

-) The data about pathways from https://reactome.org/download/current/ReactomePathways.txt, relationships between pathways from https://reactome.org/download/current/ReactomePathwaysRelation.txt and pathway-protein relations from https://reactome.org/download/current/NCBI2Reactome.txt on 24 March 2024.

-) The built knowledge graph including pathway-pathway and pathway-protein relationships.


## Scripts
The code directory contains the following scripts:

-)The script for processing data download from Reactome

-)The script for building KG and save to Neo4j Aura.


## Setup
-)conda create -n kg python=3.10 -y

-)conda activate kg

-)pip install -r requirements.txt


## Get start
python scripts/reactome/kg_reactome.py

python link_prediction_gcn/main.py --dim-latent 32 --num-layers 3 --input-size 2 --hidden-size 16 --epochs 200 --lr 0.01
python link_prediction_gcn/main.py --dim-latent 32 --num-layers 2 --input-size 2 --hidden-size 16 --epochs 2005
