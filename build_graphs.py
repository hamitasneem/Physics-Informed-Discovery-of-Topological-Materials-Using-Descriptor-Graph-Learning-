
# ================= build_graphs.py =================
# SAFE, NON-JUPYTER graph construction script
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import gc
import torch
import pandas as pd
from pymatgen.io.cif import CifParser
from pymatgen.core.periodic_table import Element
from torch_geometric.data import Data

# ---------------- THREAD LIMITS ----------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ---------------- PATHS ----------------
CIF_DIR = "cif_data/raw"
DATASET_PATH = "Files/output/qe_topo_final_ml_ready_enriched.csv"

OUT_BASE = "graph_data/pyg"
OUT_PROC = os.path.join(OUT_BASE, "processed")
OUT_STATS = os.path.join(OUT_BASE, "stats")

os.makedirs(OUT_PROC, exist_ok=True)
os.makedirs(OUT_STATS, exist_ok=True)

# ---------------- LOAD LABELS ----------------
df = pd.read_csv(DATASET_PATH)
df["target"] = (df["nu0"] > 0).astype(int)
label_map = dict(zip(df["formula"], df["target"]))

# ---------------- FEATURES ----------------
def atom_features(el):
    return [
        el.Z,                                      # atomic number
        el.X if el.X is not None else 0.0,         # electronegativity
        el.atomic_radius if el.atomic_radius else 0.0,
        el.group if el.group else 0,
        el.row if el.row else 0,
        el.is_transition_metal,
        el.is_lanthanoid,
        el.is_actinoid
    ]

def structure_to_pyg(structure, y, r_cut=6.0):
    x = torch.tensor(
        [atom_features(site.specie) for site in structure.sites],
        dtype=torch.float
    )

    edge_index, edge_attr = [], []

    for i, site in enumerate(structure.sites):
        for nn in structure.get_neighbors(site, r_cut):
            edge_index.append([i, nn.index])
            edge_attr.append([nn.nn_distance])

    if not edge_index:
        raise RuntimeError("No edges")

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([y], dtype=torch.long)
    )

# ---------------- RESUME STATE ----------------
stats_path = os.path.join(OUT_STATS, "graph_stats.csv")
fails_path = os.path.join(OUT_STATS, "failures.csv")

if os.path.exists(stats_path):
    stats = pd.read_csv(stats_path).to_dict("records")
else:
    stats = []

if os.path.exists(fails_path):
    failures = pd.read_csv(fails_path).to_dict("records")
else:
    failures = []

done = set(d["formula"] for d in stats)
failed = set(d["formula"] for d in failures)

cif_files = [f for f in os.listdir(CIF_DIR) if f.endswith(".cif")]

BATCH_SIZE = 20

# ---------------- MAIN LOOP ----------------
for start in range(0, len(cif_files), BATCH_SIZE):
    batch = cif_files[start:start+BATCH_SIZE]
    print(f"Processing batch {start} → {start+len(batch)}")

    for cif_file in batch:
        formula = cif_file.replace(".cif", "")
        if formula in done or formula in failed:
            continue

        try:
            structure = CifParser(os.path.join(CIF_DIR, cif_file)).parse_structures(primitive=True)[0]

            y = label_map.get(formula)
            if y is None:
                raise RuntimeError("Missing label")

            data = structure_to_pyg(structure, y)
            torch.save(data, os.path.join(OUT_PROC, f"{formula}.pt"))

            stats.append({
                "formula": formula,
                "n_nodes": data.x.size(0),
                "n_edges": data.edge_index.size(1),
                "target": y
            })

        except Exception as e:
            failures.append({"formula": formula, "reason": str(e)})

        del structure
        if "data" in locals():
            del data
        gc.collect()

    pd.DataFrame(stats).to_csv(stats_path, index=False)
    pd.DataFrame(failures).to_csv(fails_path, index=False)

print("✅ GRAPH CONSTRUCTION COMPLETE")
