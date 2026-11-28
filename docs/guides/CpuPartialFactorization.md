# CPU Partial Bin Factorization

The prototype was tested as a local patch on commit `d747e1127b612fea793aa47f8d85a915c0eaf167` (*generalizing factorization to N dials*), then removed from the code, configuration, and tests. CPU and MPS still support factorization for dials shared by every event in a bin.

## Benchmark

The benchmark used the full OA2024 inputs with real data, 2,406,926 MC events, 20,340,421 dials, and 4,140 bins. Both runs used the same Release binary on an Apple M4 Max with 10 threads and post-fit Hesse disabled.

```bash
gundamFitter -c "$input_config" \
  -of override/v13Prod.yaml override/noEigenPionFSI.yaml override/smoothDialResponseFloor.yaml \
  "$mode_yaml" -t 10 --backend CPU --skip-hesse -o "$output_root"
```

The `mode_yaml` override toggled partial factorization in the prototype binary. The prototype detected 50 mutually exclusive dials and 42,620 groups. The pre-fit bin contents and errors agreed within 10⁻¹⁴ after normalizing each difference by `max(1, |reference|)`.

| Partial factorization | Throughput over the first 166,332 calls | Migrad result |
| --- | ---: | --- |
| Off | 575 calls/s | Converged in 4 min 49 s after 166,332 calls |
| On | ≈604 calls/s | Stopped manually after 12 min 21 s before convergence, with 339,888 calls at the last logged checkpoint |

The throughput with partial factorization enabled was estimated by interpolating between two logged checkpoints. Migrad timings exclude data loading and initialization. The run logs, benchmark script, and prototype patch are available locally in `/tmp/gundam-oa-cpu-partial-ab.AUTXaY/`.

## Decision

The prototype evaluated about 5% more calls per second and was still running well after the baseline converged. The cause of the difference in convergence remains undetermined. Before revisiting this optimization, replay the same sequence of Migrad points in both modes, compare likelihood values and finite differences, and measure time to convergence.

After removal, the Release build succeeded and all 227 backend and minimizer tests passed.
