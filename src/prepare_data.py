"""
Download and lay out all datasets used in the paper.

Layout produced (relative to this file):

    data/hetero/<name>.npz       8 heterophilous benchmarks, from the
                                 heterophilous-graphs release (Platonov et al.)
    data/planetoid/              Cora / Citeseer / Pubmed, downloaded by PyG

The two roots are picked up by ``run_main.py`` through the environment
variables ``CF_HETERO_ROOT`` and ``CF_PLANETOID_ROOT``; if you keep the layout
produced here, no configuration is needed.

Usage
-----
    python prepare_data.py                 # everything
    python prepare_data.py --only hetero   # only the 8 npz files
"""

import os
import sys
import argparse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
HETERO_DIR = os.path.join(HERE, "data", "hetero")
PLANETOID_DIR = os.path.join(HERE, "data", "planetoid")

# Official release of Platonov et al., "A Critical Look at the Evaluation of
# GNNs on Heterophily Graphs" (NeurIPS 2022 Datasets & Benchmarks).
BASE_URL = "https://github.com/yandex-research/heterophilous-graphs/raw/main/data/{}.npz"

HETERO_FILES = [
    "chameleon_filtered", "squirrel_filtered", "actor",
    "texas", "cornell", "wisconsin",
    "tolokers", "amazon_ratings",
]

# The npz files ship ten official 60/10/30 splits each; the paper rotates over
# them (seed % 10), so one download per dataset is enough for all seeds.


def download(url, dst, retries=3):
    if os.path.exists(dst) and os.path.getsize(dst) > 1024:
        print(f"  [skip] {os.path.basename(dst)} already present "
              f"({os.path.getsize(dst)/1e6:.1f} MB)")
        return True
    for attempt in range(1, retries + 1):
        try:
            print(f"  [get ] {os.path.basename(dst)} (attempt {attempt})")
            urllib.request.urlretrieve(url, dst)
            print(f"  [ok  ] {os.path.getsize(dst)/1e6:.1f} MB")
            return True
        except Exception as e:
            print(f"  [fail] {e}")
    return False


def prepare_hetero():
    os.makedirs(HETERO_DIR, exist_ok=True)
    print(f"\nHeterophilous benchmarks -> {HETERO_DIR}")
    failed = []
    for name in HETERO_FILES:
        if not download(BASE_URL.format(name), os.path.join(HETERO_DIR, f"{name}.npz")):
            failed.append(name)
    if failed:
        print("\n[WARN] These files could not be downloaded automatically: "
              f"{', '.join(failed)}")
        print("       Grab them manually from the heterophilous-graphs repository")
        print(f"       and drop them into {HETERO_DIR}/")
        return False
    return True


def prepare_planetoid():
    print(f"\nPlanetoid (Cora / Citeseer / Pubmed) -> {PLANETOID_DIR}")
    os.makedirs(PLANETOID_DIR, exist_ok=True)
    try:
        from torch_geometric.datasets import Planetoid
    except ImportError:
        print("  [fail] torch_geometric is not installed; see requirements.txt")
        return False
    for name in ["Cora", "Citeseer", "Pubmed"]:
        try:
            d = Planetoid(root=PLANETOID_DIR, name=name)
            print(f"  [ok  ] {name}: {d[0].num_nodes} nodes, "
                  f"{d[0].num_edges} directed edges, {d.num_classes} classes")
        except Exception as e:
            print(f"  [fail] {name}: {e}")
            return False
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="all", choices=["hetero", "planetoid", "all"])
    args = p.parse_args()

    ok = True
    if args.only in ("hetero", "all"):
        ok &= prepare_hetero()
    if args.only in ("planetoid", "all"):
        ok &= prepare_planetoid()

    print("\n" + ("All datasets are ready." if ok else "Some datasets are missing."))
    print(f"  CF_HETERO_ROOT   = {HETERO_DIR}")
    print(f"  CF_PLANETOID_ROOT= {PLANETOID_DIR}")
    sys.exit(0 if ok else 1)
