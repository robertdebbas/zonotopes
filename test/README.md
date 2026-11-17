# Tests

Run from the repository root:

```bash
julia --project=. test/runtests.jl
```

The tests need no network files and take a few seconds.

- `test_abstract_round_clamp.jl`: the ReLU and round-clamp transformers on single
  neurons. Covers soundness on the concrete function, the boundary `u = C + 0.5`,
  exact half-integer endpoints, and equality of the complete constraint set with the
  convex hull of the round-clamp graph, including near-vertical hull edges.
- `test_network.jl`: the full pipeline on a small synthetic network. Checks that the
  constrained bounds are no wider than the plain zonotope bounds and that sampled
  concrete errors lie inside all of them.

The test files rely on the module loaded by `runtests.jl`, so run them through it
rather than individually.
