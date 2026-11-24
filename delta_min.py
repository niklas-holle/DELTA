"""DELTA, minimal (partial reproduction: physics rows only, no learned hosts): Putnam 4.2 -> LVMS A and LVMS A+B with the
fixed physics host H_P42 (single-track model fitted on Putnam run4_2) and the online disturbance observer (D1 table +
decaying instantaneous disturbance M).
Protocol of the Deep Dynamics parser: first section with |v_x| >= 5 m/s, 15 rows of history, horizon 50 steps (2 s).
Variants: F (constant velocities), Oracle (measured velocities integrated), N0 (host alone), E (host + D1 + M, online).
Bank angle in the rollouts of F, N0 and E: measured at the start and held over the horizon (default, no future
information); --roll: measured roll of every future row; --karte: map of the other recording at the predicted position.
The Oracle always integrates the measured future velocities with the measured roll.
Prints the mean position error at 0.6 s and 2 s and a paired circular block bootstrap (95 % intervals).
Data: deep_dynamics/data/ of the Deep Dynamics repository (github.com/linklab-uva/deep-dynamics, commit 88c85653).
Usage: python delta_min.py <path to deep-dynamics/deep_dynamics/data> [--roll | --karte]"""
import sys
import numpy as np

DT, K, H, G = 0.04, 50, 15, 9.81
# vehicle: mass and distances CG - front / rear axle from Yahya et al. 2023, Tab. III (44 % static front load); the Deep
# Dynamics config (cfgs/model/deep_dynamics_iac.yaml, VEHICLE_SPECS) uses 790 kg, lf 1.248 m, lr 1.7328 m (58 % front).
# Air density: Mitschke & Wallentowitz 2014. Tyre: shape factor C 1.3 (Pacejka 2012, ch. 4.3), curvature E = 0; load
# dependence of cornering stiffness (p_K2) and peak (p_D2) are assumptions.
M, LF, LR, RHO = 803.182, 1.6566, 1.3152, 1.226
TC, PK2, PD2 = 1.3, 2.0, -0.1
H_P42 = dict(I_z=868.18323, C_d=0.95491701, C_m1=6537.5107, C_m2=8.4202051, C_r0=462.76004, C_B=2.6800547,
             C_LA=3.6854005, b=0.41809231, mu_f=1.6967252, mu_r=1.5213629, kK_f=11.461040, kK_r=14.061869,
             G_f=-0.00079207681)
LAM_G, LAM_C, KAPPA, WARM, TAU = 1 - 1 / 2000, 1 - 1 / 1000, 50.0, 500, 0.3    # observer


def load(path, bank):
    names = [h.split('(')[0].strip() for h in open(path).readline().lstrip('#').split(',')]
    c = dict(zip(names, np.loadtxt(path, delimiter=',', skiprows=1).T))
    return dict(x=c['x'], y=c['y'], psi=c['phi'], vx=c['vx'], vy=c['vy'], r=c['omega'], delta=c['delta'],
                thr=c['throttle_ped_cmd'], brake=c['brake_ped_cmd'], th=-c['roll'] if bank else 0 * c['x'])


def tyre(a, Fz, Fz0, mu, kK):
    D = mu * (1 + PD2 * (Fz / Fz0 - 1)) * Fz
    Ky = kK * Fz0 * np.sin(2 * np.arctan(Fz / (PK2 * Fz0))) / np.sin(2 * np.arctan(1 / PK2))
    return D * np.sin(TC * np.arctan(Ky / (TC * D) * a)), D


def f(s, u, p):
    """Accelerations dv_x, dv_y, dr of the host (bank angle th, downforce, rear combined slip)."""
    vx, vy, r, th, d = s['vx'], s['vy'], s['r'], u['th'], u['delta']
    Fzf0, Fzr0 = M * G * LR / (LF + LR), M * G * LF / (LF + LR)
    n, FA = np.cos(th) + vx * r * np.sin(th) / G, 0.5 * RHO * p['C_LA'] * vx ** 2
    uu = np.maximum(np.abs(vx), 0.5)
    Fyf, _ = tyre(d - np.arctan2(r * LF + vy, uu) + p['G_f'], Fzf0 * n + FA * p['b'], Fzf0, p['mu_f'], p['kK_f'])
    Fyr, Dr = tyre(np.arctan2(r * LR - vy, uu), Fzr0 * n + FA * (1 - p['b']), Fzr0, p['mu_r'], p['kK_r'])
    # throttle command in %, brake command in kPa; brake force linear in the command (C_B in N/kPa)
    Fx = (p['C_m1'] - p['C_m2'] * vx) * np.maximum(u['thr'], 0) / 100 - np.maximum(u['brake'], 0) * p['C_B'] - p['C_r0'] - p['C_d'] * vx ** 2
    Fyr = Fyr * np.sqrt(np.maximum(1 - np.clip(Fx / np.maximum(Dr, 1e-3), -1, 1) ** 2, 0) + 1e-12)
    return ((Fx - Fyf * np.sin(d)) / M + vy * r, (Fyr + Fyf * np.cos(d)) / M - vx * r + G * np.sin(th),
            (LF * Fyf * np.cos(d) - LR * Fyr) / p['I_z'])


def euler(s, dvx, dvy, dr, th):
    c, ps = np.cos(th), s['psi']
    return dict(vx=np.maximum(s['vx'] + DT * dvx, 0.1), vy=s['vy'] + DT * dvy, r=s['r'] + DT * dr, psi=ps + DT * s['r'] / c,
                x=s['x'] + DT * (s['vx'] * np.cos(ps) - s['vy'] * c * np.sin(ps)), y=s['y'] + DT * (s['vx'] * np.sin(ps) + s['vy'] * c * np.cos(ps)))


def cell(vx, r):                                                 # 7 speed x 4 |v_x r| bands
    return np.searchsorted([10, 20, 30, 40, 50, 60], vx, 'right') * 4 + np.searchsorted([2, 6, 12], np.abs(vx * r), 'right')


def observer(S, R, E, c):
    """Per start: D1 table (28 cells x 3) and mean of the last 5 residuals, from the residuals known before the start
    (residual of row j is known at row j + 1); outliers |e| >= 30 / 30 / 3 are skipped. Of the last 5 accepted residuals
    only those at most 6 rows old count; if none, there is no instantaneous term. Warm-up: the table is zero until
    500 residuals are known; the instantaneous term (rec) acts from the first known residual on."""
    ok = (np.abs(E) < [30, 30, 3]).all(1)
    R, E, c = R[ok], E[ok], c[ok]
    Gs, Gn, Cs, Cn = np.zeros(3), 0.0, np.zeros((28, 3)), np.zeros(28)
    tab, rec = np.zeros((len(S), 28, 3)), np.full((len(S), 3), np.nan)
    j = 0
    for i, s in enumerate(S):
        while j < len(R) and R[j] + 1 <= s:
            Gs, Gn = LAM_G * Gs + E[j], LAM_G * Gn + 1
            Cs[c[j]], Cn[c[j]] = LAM_C * Cs[c[j]] + E[j], LAM_C * Cn[c[j]] + 1
            j += 1
        if j >= WARM:
            tab[i] = (Cs + KAPPA * Gs / Gn) / (Cn + KAPPA)[:, None]
        last = slice(max(j - 5, 0), j)
        if (R[last] >= s - 6).any():
            rec[i] = E[last][R[last] >= s - 6].mean(0)
    return tab, rec


def rollout(d, S, p, kind, tab=None, rec=None, theta=None):
    """Positions (n, K, 2) after steps 1..K from the measured start states; kind F, Oracle or host (+ table, rec = E).
    Bank angle of step t: measured at row S + t, or theta(S, t, state) if given."""
    s, a = {k: d[k][S] for k in ('x', 'y', 'psi', 'vx', 'vy', 'r')}, np.arange(len(S))
    d0 = np.nan_to_num(rec - tab[a, cell(s['vx'], s['r'])]) if tab is not None else 0
    out = np.zeros((len(S), K, 2))
    for t in range(K):
        u = {k: d[k][S + t] for k in ('delta', 'thr', 'brake', 'th')}
        if theta is not None:
            u['th'] = theta(S, t, s)
        if kind == 'Oracle':
            s.update({k: d[k][S + t] for k in ('vx', 'vy', 'r')})
        corr = tab[a, cell(s['vx'], s['r'])] + d0 * np.exp(-t * DT / TAU) if tab is not None else np.zeros((len(S), 3))
        s = euler(s, *(f(s, u, p) if kind == 'host' else (0, 0, 0)), u['th'])
        s.update({k: s[k] + DT * corr[:, i] for i, k in enumerate(('vx', 'vy', 'r'))})
        out[:, t] = np.stack([s['x'], s['y']], 1)
    return out


def source_loss(d, p=H_P42):
    """Fit criterion: mean position error of the host alone after 2 s over every start of the source recording
    (H_P42 on Putnam run4_2: 0.691 m)."""
    S = np.arange(len(d['x']) - K)
    return np.linalg.norm(rollout(d, S, p, 'host')[:, -1] - np.stack([d['x'][S + K], d['y'][S + K]], 1), axis=1).mean()


def bank_map(src, d, cell=5.0):
    """Bank angle of recording d from another recording src: mean per 5 m cell (|v_x| >= 5), else the occupied 3 x 3
    neighbourhood, else the start value; returns theta(S, t, state) for the rollout (step 0: measured at the start)."""
    m = np.abs(src['vx']) >= 5
    x0, y0 = src['x'][m].min() - 50, src['y'][m].min() - 50
    ix, iy = ((src['x'][m] - x0) // cell).astype(int), ((src['y'][m] - y0) // cell).astype(int)
    S_, N_ = np.zeros((ix.max() + 12, iy.max() + 12)), np.zeros((ix.max() + 12, iy.max() + 12))
    np.add.at(S_, (ix, iy), src['th'][m])
    np.add.at(N_, (ix, iy), 1)
    mean, occ = S_ / np.maximum(N_, 1), (N_ > 0).astype(float)
    nb = lambda A: sum(np.pad(A, 1)[1 + a:1 + a + A.shape[0], 1 + b:1 + b + A.shape[1]] for a in (-1, 0, 1) for b in (-1, 0, 1))
    m3, o3 = nb(mean * occ) / np.maximum(nb(occ), 1), nb(occ)

    def theta(S, t, s):
        if t == 0:
            return d['th'][S]
        jx = np.clip(((s['x'] - x0) // cell).astype(int), 0, occ.shape[0] - 1)
        jy = np.clip(((s['y'] - y0) // cell).astype(int), 0, occ.shape[1] - 1)
        return np.where(occ[jx, jy] > 0, mean[jx, jy], np.where(o3[jx, jy] > 0, m3[jx, jy], d['th'][S]))
    return theta


def stream(d, p=H_P42, theta=None):
    """Errors (n, 2) at 0.6 s and 2 s of every variant on one recording, observer learning online from its first row;
    theta: bank angle in the rollouts of F, N0 and E (None: measured roll of every row; the Oracle always uses it)."""
    i0 = int(np.argmax(np.abs(d['vx']) >= 5))
    i1 = i0 + int(np.argmax(np.r_[np.abs(d['vx'][i0:]) < 5, True]))
    S, R = np.arange(i0 + H - 1, i1 - K - 1), np.arange(i0 + H - 1, i1 - 1)   # starts: 15 rows of history, K + 1 of future
    nx = euler({k: d[k][R] for k in ('x', 'y', 'psi', 'vx', 'vy', 'r')}, *f({k: d[k][R] for k in ('vx', 'vy', 'r')},
               {k: d[k][R] for k in ('delta', 'thr', 'brake', 'th')}, p), d['th'][R])
    E = np.stack([(d[k][R + 1] - nx[k]) / DT for k in ('vx', 'vy', 'r')], 1)
    tab, rec = observer(S, R, E, cell(d['vx'][R], d['r'][R]))
    gt = np.stack([d['x'][S[:, None] + np.arange(1, K + 1)], d['y'][S[:, None] + np.arange(1, K + 1)]], 2)
    P = {'F': rollout(d, S, p, 'F', theta=theta), 'Oracle': rollout(d, S, p, 'Oracle'), 'N0': rollout(d, S, p, 'host', theta=theta),
         'E': rollout(d, S, p, 'host', tab, rec, theta)}
    return {v: np.linalg.norm(x - gt, axis=2)[:, [14, 49]] for v, x in P.items()}


def boot(parts, B=2000, L=750, seed=0):
    """Paired circular moving-block bootstrap of the mean errors; blocks drawn per recording, pooled by start counts."""
    rng, tot = np.random.default_rng(seed), sum(len(p['F']) for p in parts)
    out = {v: np.zeros((B, 2)) for v in parts[0]}
    for p in parts:
        n = len(p['F'])
        Lb = min(L, n)
        nb = -(-n // Lb)
        bs = rng.integers(0, n, size=(B, nb))
        for v, e in p.items():
            c = np.concatenate([np.zeros((1, 2)), np.cumsum(np.concatenate([e, e[:Lb]]), 0)])
            out[v] += (c[bs[:, :-1] + Lb] - c[bs[:, :-1]]).sum(1) + c[bs[:, -1] + n - (nb - 1) * Lb] - c[bs[:, -1]]
    return {v: x / tot for v, x in out.items()}


def bank_start(d):
    """Bank angle measured at the start and held over the horizon."""
    return lambda S, t, s: d['th'][S]


if __name__ == '__main__':
    D = {r: load(f'{sys.argv[1]}/LVMS_23_01_04_{r}.csv', bank=True) for r in 'AB'}
    mode = 'karte' if '--karte' in sys.argv else 'roll' if '--roll' in sys.argv else 'start'
    th = {'start': lambda r: bank_start(D[r]), 'roll': lambda r: None, 'karte': lambda r: bank_map(D['B' if r == 'A' else 'A'], D[r])}[mode]
    lv = {r: stream(D[r], theta=th(r)) for r in 'AB'}
    assert (len(lv['A']['F']), len(lv['B']['F'])) == (13356, 9982), 'start counts differ from the protocol'
    print({'start': 'Bank angle: measured at the start, held over the horizon', 'roll': 'Bank angle: measured roll of every future row',
           'karte': 'Bank angle: map of the other recording'}[mode] + ' (Oracle: measured roll)')
    print('Mean position error [m] at 0.6 s / 2 s (relative to F); L = (F - E) / (F - Oracle)')
    for name, recs in (('S1 Putnam 4.2 -> LVMS A', 'A'), ('S2 Putnam 4.2 -> LVMS A+B', 'AB')):
        m = {v: np.concatenate([lv[r][v] for r in recs]).mean(0) for v in lv['A']}
        print(f'\n{name} ({sum(len(lv[r]["F"]) for r in recs)} starts)')
        for v, x in m.items():
            print(f'  {v:7s} {x[0]:.3f} ({100 * (x[0] / m["F"][0] - 1):+4.0f} %)   {x[1]:.3f} ({100 * (x[1] / m["F"][1] - 1):+4.0f} %)')
        print(f'  L(E)    {100 * (m["F"][0] - m["E"][0]) / (m["F"][0] - m["Oracle"][0]):.0f} %   {100 * (m["F"][1] - m["E"][1]) / (m["F"][1] - m["Oracle"][1]):.0f} %')
        b = boot([lv[r] for r in recs])
        print('  Paired bootstrap: E against F / N0, relative difference [95 % interval], 0.6 s | 2 s')
        for ref in ('F', 'N0'):
            rel = 100 * (b['E'] / b[ref] - 1)
            lo, hi = np.percentile(rel, [2.5, 97.5], axis=0)
            print(f'  E - {ref:3s} ' + ' | '.join(f'{100 * (m["E"][j] / m[ref][j] - 1):+.1f} % [{lo[j]:+.1f}; {hi[j]:+.1f}]' for j in (0, 1)))
