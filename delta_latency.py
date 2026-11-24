"""DELTA, run time: batched rollout of the physics model + DELTA in PyTorch and its latency over the number of candidates.
One control cycle = B candidate input sequences rolled out over K = 50 steps (2 s) from the same measured state; all
candidates share one disturbance table and look it up at their own predicted states.
First checks the PyTorch rollout against delta_min.rollout (float64), then times it (float32): eager and as a CUDA graph
on the GPU, eager on the CPU, plus the observer update per measurement.
Usage: python delta_latency.py <recording.csv from deep-dynamics/deep_dynamics/data> [--max N]
--max N: continue the CUDA-graph rows in powers of two up to B = 2^N (default 14, i.e. 16 384)."""
import sys
import time
import numpy as np
import torch
import delta_min as dm

K, DT = dm.K, dm.DT
VB, AB = [10, 20, 30, 40, 50, 60], [2, 6, 12]                        # cell bounds as in delta_min.cell


def tyre(a, Fz, Fz0, mu, kK):
    D = mu * (1 + dm.PD2 * (Fz / Fz0 - 1)) * Fz
    Ky = kK * Fz0 * torch.sin(2 * torch.atan(Fz / (dm.PK2 * Fz0))) / np.sin(2 * np.arctan(1 / dm.PK2))
    return D * torch.sin(dm.TC * torch.atan(Ky / (dm.TC * D) * a)), D


BOUNDS = {}


def bounds(dtype, dev):
    """Cell bounds as tensors, created once per dtype and device (no host-to-GPU copy inside a CUDA graph)."""
    if (dtype, dev) not in BOUNDS:
        BOUNDS[dtype, dev] = torch.tensor(VB, dtype=dtype, device=dev), torch.tensor(AB, dtype=dtype, device=dev)
    return BOUNDS[dtype, dev]


def rollout(s0, u, th, tab, rec, p=dm.H_P42):
    """Positions (B, K, 2). s0: (B, 6) x, y, psi, vx, vy, r; u: (B, K, 3) delta, throttle, brake; th: (B,) bank angle held;
    tab: (28, 3) shared or (B, 28, 3); rec: (B, 3) mean of the last residuals (NaN: no instantaneous term)."""
    x, y, psi, vx, vy, r = s0.unbind(1)
    vb, ab = bounds(s0.dtype, s0.device)
    rows = torch.arange(len(s0), device=s0.device)

    def look():                                                       # table at the predicted state of every candidate
        c = torch.bucketize(vx, vb, right=True) * 4 + torch.bucketize((vx * r).abs(), ab, right=True)
        return tab[c] if tab.dim() == 2 else tab[rows, c]
    d0 = torch.nan_to_num(rec - look())
    Fzf0, Fzr0 = dm.M * dm.G * dm.LR / (dm.LF + dm.LR), dm.M * dm.G * dm.LF / (dm.LF + dm.LR)
    c, sn = torch.cos(th), torch.sin(th)
    out = []
    for t in range(K):
        d, thr, brk = u[:, t].unbind(1)
        corr = look() + d0 * float(np.exp(-t * DT / dm.TAU))
        n, FA, uu = c + vx * r * sn / dm.G, 0.5 * dm.RHO * p['C_LA'] * vx ** 2, vx.abs().clamp(min=0.5)
        Fyf, _ = tyre(d - torch.atan2(r * dm.LF + vy, uu) + p['G_f'], Fzf0 * n + FA * p['b'], Fzf0, p['mu_f'], p['kK_f'])
        Fyr, Dr = tyre(torch.atan2(r * dm.LR - vy, uu), Fzr0 * n + FA * (1 - p['b']), Fzr0, p['mu_r'], p['kK_r'])
        Fx = (p['C_m1'] - p['C_m2'] * vx) * thr.clamp(min=0) / 100 - brk.clamp(min=0) * p['C_B'] - p['C_r0'] - p['C_d'] * vx ** 2
        Fyr = Fyr * torch.sqrt((1 - (Fx / Dr.clamp(min=1e-3)).clamp(-1, 1) ** 2).clamp(min=0) + 1e-12)
        dvx = (Fx - Fyf * torch.sin(d)) / dm.M + vy * r
        dvy = (Fyr + Fyf * torch.cos(d)) / dm.M - vx * r + dm.G * sn
        dr = (dm.LF * Fyf * torch.cos(d) - dm.LR * Fyr) / p['I_z']
        x, y, psi = x + DT * (vx * torch.cos(psi) - vy * c * torch.sin(psi)), y + DT * (vx * torch.sin(psi) + vy * c * torch.cos(psi)), psi + DT * r / c
        vx, vy, r = (vx + DT * dvx).clamp(min=0.1) + DT * corr[:, 0], vy + DT * (dvy + corr[:, 1]), r + DT * (dr + corr[:, 2])
        out.append(torch.stack([x, y], 1))
    return torch.stack(out, 1)


def stream(d, p=dm.H_P42):
    """Starts, table and instantaneous term per start as in delta_min.stream (first section with |v_x| >= 5 m/s)."""
    i0 = int(np.argmax(np.abs(d['vx']) >= 5))
    i1 = i0 + int(np.argmax(np.r_[np.abs(d['vx'][i0:]) < 5, True]))
    S, R = np.arange(i0 + dm.H - 1, i1 - K - 1), np.arange(i0 + dm.H - 1, i1 - 1)
    nx = dm.euler({k: d[k][R] for k in ('x', 'y', 'psi', 'vx', 'vy', 'r')}, *dm.f({k: d[k][R] for k in ('vx', 'vy', 'r')},
                  {k: d[k][R] for k in ('delta', 'thr', 'brake', 'th')}, p), d['th'][R])
    E = np.stack([(d[k][R + 1] - nx[k]) / DT for k in ('vx', 'vy', 'r')], 1)
    return S, R, E, *dm.observer(S, R, E, dm.cell(d['vx'][R], d['r'][R]))


def tensors(d, S, tab, rec, dtype, dev):
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=dev)
    s0 = t(np.stack([d[k][S] for k in ('x', 'y', 'psi', 'vx', 'vy', 'r')], 1))
    u = t(np.stack([d[k][S[:, None] + np.arange(K)] for k in ('delta', 'thr', 'brake')], 2))
    return s0, u, t(d['th'][S]), t(tab), t(rec)


def timed(fn, sync, budget=2.0, nmin=20, nmax=200):
    for _ in range(3):
        fn()
    ts, t_all = [], time.perf_counter()
    while len(ts) < nmax and (len(ts) < nmin or time.perf_counter() - t_all < budget):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts)), 1e3 * float(np.percentile(ts, 95))


def graphed(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


if __name__ == '__main__':
    d = dm.load(sys.argv[1], bank=False)                              # flat: the roll channel of Putnam run4_2 is not usable
    S, R, E, tab, rec = stream(d)
    ref = dm.rollout(d, S, dm.H_P42, 'host', tab, rec, dm.bank_start(d))
    got = rollout(*tensors(d, S, tab, rec, torch.float64, 'cpu')).numpy()
    print(f'Check against delta_min.rollout ({len(S)} starts, float64): max deviation {np.abs(got - ref).max():.1e} m')
    i = len(S) - 1                                                    # last start: table after the whole section
    rng = np.random.default_rng(0)
    print(f'\nLatency of one cycle, K = {K} steps, float32 (median / p95)')
    print(f'{"device":18s} {"B":>6s} {"ms":>8s} {"p95":>8s} {"Hz":>7s} {"ms / candidate":>15s}')
    gpu = torch.cuda.is_available()
    emax = int(sys.argv[sys.argv.index('--max') + 1]) if '--max' in sys.argv else 14
    big = tuple(2 ** k for k in range(15, emax + 1))
    runs = [('cpu', 'CPU, eager', (1, 16, 256))]
    runs += [('cuda', 'GPU, eager', (1, 1024, 16384)), ('cuda', 'GPU, CUDA graph', (1, 16, 256, 1024, 4096, 16384) + big)] if gpu else []
    for dev, name, Bs in runs:
        sync = torch.cuda.synchronize if dev == 'cuda' else (lambda: None)
        for B in Bs:
            s0, u, th, tb, rc = tensors(d, np.full(B, S[i]), tab[i], np.repeat(rec[[i]], B, 0), torch.float32, dev)
            noise = rng.normal(0, [0.005, 2.0, 0.0], (B, K, 3))        # candidates differ in steering and throttle
            u = u + torch.as_tensor(noise, dtype=torch.float32, device=dev)
            with torch.no_grad():
                fn = lambda: rollout(s0, u, th, tb, rc)
                ms, p95 = timed(graphed(fn) if 'graph' in name else fn, sync)
            print(f'{name:18s} {B:6d} {ms:8.2f} {p95:8.2f} {1e3 / ms:7.0f} {ms / B:15.5f}')
    if gpu:
        print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)
    # observer update per measurement (NumPy): one residual into the global estimate and into its cell
    st = {'Gs': np.zeros(3), 'Gn': 1.0, 'Cs': np.zeros((28, 3)), 'Cn': np.ones(28)}
    e, c = E[-1], int(dm.cell(d['vx'][R[-1:]], d['r'][R[-1:]])[0])

    def update():
        st['Gs'], st['Gn'] = dm.LAM_G * st['Gs'] + e, dm.LAM_G * st['Gn'] + 1
        st['Cs'][c], st['Cn'][c] = dm.LAM_C * st['Cs'][c] + e, dm.LAM_C * st['Cn'][c] + 1
    ms, p95 = timed(update, lambda: None, nmax=2000)
    print(f'Observer update per measurement (NumPy): {ms:.4f} ms (p95 {p95:.4f} ms)')
