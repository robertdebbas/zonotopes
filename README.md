# Quantized Zonotope Verification

Code for the paper *Zonotopic Analysis of Quantization Error in Neural Networks*.

Given a ReLU network, its fixed-point quantized counterpart and a box of inputs, the
code computes sound bounds on the quantization error `f_QNN(x) − f_DNN(x)`. Both
networks are propagated through zonotope transformers (ReLU for the original network,
a combined round-clamp transformer for the quantized one) and the output zonotopes are
subtracted.

Three variants are available:

| Variant | What it does |
|---|---|
| Zonotope | Plain zonotope propagation, no LP |
| Constrained (light) | Adds two box constraints per unstable neuron and solves LPs with HiGHS. This is the main method of the paper. |
| Constrained (complete) | Also adds the remaining convex-hull constraints of the round-clamp relaxation |

## Installation

Tested with Julia 1.12 (the versions in `Manifest.toml` are pinned for it).

```bash
git clone --recurse-submodules https://github.com/robertdebbas/zonotopes.git
cd zonotopes
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`ModelVerification.jl/` is a small local package that defines the `Network` and
`Layer` types. `QEBVerif/` is a git submodule and is only needed for the ACAS Xu
weights and the DRA-Sym baseline.

## Quick check

Neither command needs network files.

```bash
julia --project=. test/runtests.jl
julia --project=. examples/small_example.jl
```

The example bounds the quantization error of a small synthetic network and prints:

```
zonotope:    [-0.2040218505859376] [0.24231811523437488]
constrained: [-0.1730312500000001] [0.19828124999999988]
```

## Usage

The repository is not a registered package, so load the module with `include`
(see `examples/small_example.jl` for a complete script):

```julia
include("src/QuantizedZonotopeVerification.jl")
using .QuantizedZonotopeVerification
using LazySets, LinearAlgebra

net = load_acasxu_network_from_json("networks/acasxu_1_1.json")

# (signedness tag, total bits, fractional bits)
quant_config = Dict(
    :input       => ("pm", 8, 8),
    :weights     => ("pm", 6, 4),
    :biases      => ("pm", 6, 4),
    :activations => ("p",  6, 4),
)

input_zono = Zonotope(fill(-0.3, 5), 0.01 * Matrix(1.0I, 5, 5))

# Plain zonotope: returns the error zonotope
error_zono = quantization_error_zonotope(net, quant_config, input_zono)
box = overapproximate(error_zono, Hyperrectangle)

# Constrained (light): returns lower and upper bound vectors
lb, ub = quantization_error_zonotope(net, quant_config, input_zono;
             constrained=true, full_constraints=false, diagonal_constraints=false)

# Constrained (complete)
lb, ub = quantization_error_zonotope(net, quant_config, input_zono; constrained=true)
```

Weights, biases and inputs are quantized as signed fixed-point values and hidden
activations as unsigned ones; the signedness tag in `quant_config` is not read.
Rounding is to nearest with ties to even. The constrained variants expect a box-shaped
input zonotope with one generator per input dimension.

## Reproducing the experiments

All commands are run from the repository root.

**1. ACAS Xu networks.** The benchmarks read `networks/acasxu_<i>_<j>.json`. Generate
these files from the weights in the QEBVerif submodule (requires `h5py`):

```bash
python benchmarks/convert_networks.py
```

**2. Zonotope methods.** 150 experiments: 10 networks × Q ∈ {6, 8, 10} × r ∈ {3, 6, 13, 19, 26},
a few minutes in total.

```bash
julia --project=. benchmarks/collect_data.jl
```

This overwrites `results/data/benchmark_results.csv`. The bound values are
deterministic; the timings depend on the machine. Each LP has a 2-second limit, and an
LP that does not finish falls back to the plain zonotope bound for that output, so a
heavily loaded machine can produce looser constrained bounds.

**3. DRA-Sym baseline.** Needs a Python environment with TensorFlow 2, `gurobipy` and
a Gurobi license (see `QEBVerif/README.md`).

```bash
python benchmarks/collect_drasym_official.py
```

This writes `results/data/drasym_results.csv`.

**4. Figures** (requires `pandas` and `matplotlib`), written to `results/figures/`:

```bash
python benchmarks/plot_results.py
```

**5. Ablations.** Both scripts print their tables to the terminal.

```bash
julia --project=. ablations/light_vs_complete.jl   # light vs. complete constraints
julia --project=. ablations/mink_vs_shared.jl      # independent vs. shared input symbols
```

## Results

ACAS Xu, 150 experiments, on an Apple M5 CPU. ODiff is the mean width of the error
interval over the 5 outputs. The per-experiment values behind these tables are in
`results/data/benchmark_results.csv` and `results/data/drasym_results.csv`.

| Method | Avg ODiff | Avg time (s) |
|---|---|---|
| Zonotope | 361.1 | 0.002 |
| Constrained (light) | 230.0 | 0.382 |
| Constrained (complete) | 230.0 | 0.607 |
| DRA-Sym (QEBVerif) | 145.3 | 0.564 |

Small radii only (r = 3, 6):

| Method | Avg ODiff | Avg time (s) |
|---|---|---|
| Zonotope | 4.99 | 0.001 |
| Constrained (light) | 3.03 | 0.132 |
| Constrained (complete) | 3.02 | 0.244 |
| DRA-Sym (QEBVerif) | 11.69 | 0.454 |

At small radii the constrained variants give tighter bounds than DRA-Sym in less time;
at larger radii DRA-Sym is tighter.

## Repository layout

```
src/
  QuantizedZonotopeVerification.jl   module and exports
  core/abstractions.jl               abstract_relu, abstract_round_clamp
  core/propagation.jl                network propagation, constraints, LP bounds
  core/quantization.jl               quantize_tensor, quantize_zonotope
  core/sampling.jl                   concrete forward passes, sampled error range
  utils/                             box/zonotope conversions, network loading
examples/small_example.jl            small synthetic example
test/                                unit and end-to-end tests
benchmarks/                          benchmark, baseline, conversion and plotting scripts
ablations/                           the two ablation studies
results/                             output CSVs and figures
ModelVerification.jl/                local package with the Network and Layer types
QEBVerif/                            baseline (git submodule)
```

## License

The code in this repository is released under the MIT License (see `LICENSE`).

QEBVerif, linked as a git submodule, is a separate project by its own authors and is
not covered by this license. It is only needed for the ACAS Xu weights and the DRA-Sym
baseline; see its repository for its terms of use.

## Citation

```bibtex
@misc{debbas2026zonotopic,
  title  = {Zonotopic Analysis of Quantization Error in Neural Networks},
  author = {Debbas, Robert and Goubault, Eric and Putot, Sylvie},
  year   = {2026},
  note   = {arXiv preprint}
}
```
