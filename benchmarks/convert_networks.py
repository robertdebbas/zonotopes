"""
Convert the ACAS Xu weights shipped with QEBVerif (Keras .h5) to the JSON format
read by load_acasxu_network_from_json.

Run from anywhere (requires h5py):
  python benchmarks/convert_networks.py

Reads  QEBVerif/benchmark/acasxu/ACASXU_run2a_<i>_<j>_batch_2000.h5
Writes networks/acasxu_<i>_<j>.json

JSON layout: {"layer1": {"W": [...], "b": [...]}, ..., "layer7": {...}}, where
W[k] holds the weights from input k to every neuron of the layer.
"""

import glob, json, os, re, sys

import h5py

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H5_DIR    = os.path.join(REPO_ROOT, "QEBVerif", "benchmark", "acasxu")
OUT_DIR   = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "networks")


def layer_index(name):
    """quantized_dense -> 0, quantized_dense_3 -> 3"""
    m = re.search(r"quantized_dense_(\d+)", name)
    return int(m.group(1)) if m else 0


def convert(h5_path):
    kernels, biases = {}, {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            target = kernels if obj.ndim == 2 else biases
            target[layer_index(name)] = obj[()].tolist()

    with h5py.File(h5_path, "r") as f:
        f.visititems(visit)

    # Keras numbers layers per session, so indices need not start at 0: renumber from 1.
    return {f"layer{n}": {"W": kernels[k], "b": biases[k]}
            for n, k in enumerate(sorted(kernels), start=1)}


if __name__ == "__main__":
    paths = sorted(glob.glob(os.path.join(H5_DIR, "ACASXU_run2a_*_batch_2000.h5")))
    if not paths:
        sys.exit(f"No .h5 files found in {H5_DIR}. Is the QEBVerif submodule checked out?")

    os.makedirs(OUT_DIR, exist_ok=True)
    for path in paths:
        i, j = re.search(r"run2a_(\d+)_(\d+)_batch", path).groups()
        out = os.path.join(OUT_DIR, f"acasxu_{i}_{j}.json")
        with open(out, "w") as f:
            json.dump(convert(path), f)
    print(f"Wrote {len(paths)} networks to {OUT_DIR}")
