# DELTA: minimal reproduction

Companion code to *DELTA: Online Disturbance Learning for Cross-Track Vehicle Dynamics Prediction in Autonomous Racing*.

One self-contained file, `delta_min.py` (193 lines, NumPy only), reproduces the physics rows of the cross-track result
Putnam 4.2 → LVMS A and LVMS A+B on the RACECAR data:

- a single-track physics model fitted on Putnam run4_2 (parameters in the file, nothing is fitted at run time);
- DELTA, the online disturbance observer: a table of one-step velocity residuals over speed × lateral acceleration plus a
  decaying instantaneous term. It learns only from residuals that are already known when a prediction starts.

A second file, `delta_latency.py` (144 lines, PyTorch), measures the run time of the same rollout for many candidate
input sequences at once (see [Run time](#run-time)).

Not included: the learned models (DDM, DynSSM), the other data sets and the ablations.

## Data

The recordings come from the `deep_dynamics/data/` folder of the Deep Dynamics repository
([linklab-uva/deep-dynamics](https://github.com/linklab-uva/deep-dynamics), commit `88c85653`). The script reads
`LVMS_23_01_04_A.csv` and `LVMS_23_01_04_B.csv`.

## Usage

```
pip install numpy
python delta_min.py <path to deep-dynamics>/deep_dynamics/data
```

Runtime: a few seconds on a laptop CPU.

Options for the bank angle in the rollouts (default: measured at the start and held, no future information):
`--roll` uses the measured roll of every future row, `--karte` a map built from the other LVMS recording.

## Protocol

- First contiguous section with |v_x| ≥ 5 m/s, as in the Deep Dynamics parser; a start needs 15 rows of history and
  51 rows of future. This gives 13 356 starts on LVMS A and 9 982 on LVMS B (checked by the script).
- Horizon 50 steps (2 s at 25 Hz); mean position error at 0.6 s and 2 s.
- `F`: constant velocities. `Oracle`: measured velocities integrated. `N0`: physics model alone. `E`: physics model + DELTA.
- Paired circular moving-block bootstrap (blocks of 750 starts, 2000 replicates, seed 0), 95 % intervals.
- `source_loss(d)` recomputes the fit criterion of the physics model on Putnam run4_2 (0.691 m).

## Expected output

```
Bank angle: measured at the start, held over the horizon (Oracle: measured roll)
Mean position error [m] at 0.6 s / 2 s (relative to F); L = (F - E) / (F - Oracle)

S1 Putnam 4.2 -> LVMS A (13356 starts)
  F       0.178 (  +0 %)   0.971 (  +0 %)
  Oracle  0.157 ( -12 %)   0.504 ( -48 %)
  N0      0.196 ( +10 %)   1.469 ( +51 %)
  E       0.161 ( -10 %)   0.584 ( -40 %)
  L(E)    81 %   83 %
  Paired bootstrap: E against F / N0, relative difference [95 % interval], 0.6 s | 2 s
  E - F   -9.5 % [-16.6; -4.0] | -39.8 % [-49.6; -29.1]
  E - N0  -17.6 % [-22.6; -10.7] | -60.3 % [-65.2; -52.1]

S2 Putnam 4.2 -> LVMS A+B (23338 starts)
  F       0.178 (  +0 %)   1.098 (  +0 %)
  Oracle  0.135 ( -24 %)   0.422 ( -62 %)
  N0      0.184 (  +3 %)   1.263 ( +15 %)
  E       0.144 ( -20 %)   0.587 ( -47 %)
  L(E)    80 %   76 %
  Paired bootstrap: E against F / N0, relative difference [95 % interval], 0.6 s | 2 s
  E - F   -19.6 % [-25.5; -12.7] | -46.6 % [-52.2; -38.9]
  E - N0  -22.0 % [-25.2; -17.9] | -53.6 % [-58.3; -47.2]
```

## Run time

```
pip install numpy torch
python delta_latency.py <path to deep-dynamics>/deep_dynamics/data/Putnam_park2023_run4_2.csv [--max N]
```

`delta_latency.py` rolls out the physics model + DELTA for B candidate input sequences from the same measured state
(one control cycle of a sampling-based MPC). All candidates share one disturbance table and look it up at their own
predicted states. The script

1. checks the PyTorch rollout against `delta_min.rollout` on every start of the recording (float64; the maximum deviation
   is about 1e-13 m);
2. times one cycle of 50 steps in float32: as a CUDA graph and eagerly on the GPU, eagerly on the CPU;
3. times the observer update per measurement (NumPy).

Any recording of the data folder works; the run time does not depend on the data. The recording is loaded without the
bank angle, because the roll channel of Putnam run4_2 is not usable.

### Expected output

Measured on a laptop:

- CPU: AMD Ryzen AI 7 350 (8 cores, 16 threads);
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU;
- Windows 11, Python 3.12, NumPy 2.4.3, PyTorch 2.11.0 (CUDA 12.8).

```
Check against delta_min.rollout (11441 starts, float64): max deviation 1.1e-13 m

Latency of one cycle, K = 50 steps, float32 (median / p95)
device                  B       ms      p95      Hz  ms / candidate
CPU, eager              1    49.92    62.81      20        49.92490
CPU, eager             16    48.78    57.64      20         3.04884
CPU, eager            256    56.75    60.20      18         0.22169
GPU, eager              1   198.04   218.51       5       198.04195
GPU, eager           1024   203.29   217.74       5         0.19852
GPU, eager          16384   202.54   227.84       5         0.01236
GPU, CUDA graph         1     9.75    11.07     103         9.75265
GPU, CUDA graph        16    10.25    13.02      98         0.64033
GPU, CUDA graph       256    10.01    11.07     100         0.03912
GPU, CUDA graph      1024    10.06    11.25      99         0.00982
GPU, CUDA graph      4096    10.47    12.22      96         0.00256
GPU, CUDA graph     16384    10.00    11.93     100         0.00061
GPU: NVIDIA GeForce RTX 5060 Laptop GPU | torch 2.11.0+cu128
Observer update per measurement (NumPy): 0.0058 ms (p95 0.0094 ms)
```

The check line is exact up to rounding. The times depend on the hardware. Between runs on this laptop, the CUDA-graph
times varied by a few percent; the eager times are bound by the CPU and varied by up to about 25 %.
Without a CUDA device, only the CPU rows and the observer update are printed.

Notes:

- **The numbers need a CUDA graph.** The rollout consists of many small GPU operations, so the time is set by the number
  of kernel launches, not by the batch size. Run eagerly, one cycle takes 170 to 220 ms on the same GPU, for any B. The
  script prints both. In this regime the eager time is bound by the CPU, which issues the launches.
- **The time is nearly constant in B** up to 16 384 candidates, for the same reason.
- **On the CPU, use NumPy.** The eager PyTorch rollout on the CPU takes 40 to 56 ms for one candidate; the NumPy rollout
  in `delta_min.py` is faster there.
- This file is a compact re-implementation for the repository. The run times reported in the paper were measured with the
  full evaluation pipeline on an RTX 5060 Ti (9.4 ms for one candidate, about 10 ms for 16 384); the laptop numbers above
  are in the same range, but they are not the same measurement.
- The learned models compared in the paper are not part of this repository.
- **Side note, beyond 16 384 candidates:** `--max N` continues the CUDA-graph rows in powers of two up to B = 2^N. On the
  same laptop GPU (8 GB), the time stays near 10 ms up to about 32 000 candidates and then grows linearly, at roughly
  0.2 µs per candidate: 131 072 candidates take about 19 ms (54 Hz), one million about 177 ms. Memory ends between 4 and
  8 million candidates (`--max 22` still fits). This covers the rollout only, without cost evaluation, and is not part of
  the paper.

## Vehicle data

Mass and axle distances from Yahya et al. 2023, Tab. III (803.2 kg, l_f 1.657 m, l_r 1.315 m). Sources of the remaining
constants are given as comments in the file.

## License

MIT, see `LICENSE`.
