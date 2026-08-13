# Numeric feature, encoder, and loss contract

This page is the authoritative map from stored IceCube quantities to the
numbers seen by IceMix. Read it before changing feature order, units, detector
normalization, Fourier scales, the relative-attention metric, output layout, or
the joint-loss coefficient. The equations below are **VERIFIED-STATIC** against
this GraphNeT checkout and IceMix configuration. The historical motivation and
tuning provenance of the numeric constants remain **UNVERIFIED-CLUSTER** unless
stated otherwise.

## Ordered pulse features

GraphNeT requests this seven-column order from each pulse table:

| Index | Stored field | GraphNeT transform before IceMix | IceMix use at baseline `n_features=6` |
|---:|---|---|---|
| 0 | `dom_x` | $x' = x / 500$ | Fourier encoding and pairwise geometry |
| 1 | `dom_y` | $y' = y / 500$ | Fourier encoding and pairwise geometry |
| 2 | `dom_z` | $z' = z / 500$ | Fourier encoding and pairwise geometry |
| 3 | `dom_time` | $t' = (t - 10{,}000) / 30{,}000$ | Fourier encoding and pairwise geometry |
| 4 | `charge` | $q' = \log_{10}(q)$ | Fourier encoding |
| 5 | `rde` | $r' = (r - 1.25) / 0.25$ | Fourier encoding |
| 6 | `pmt_area` | $a' = a / 0.05$ | Compatibility column; not Fourier-encoded by baseline |

The table names and formulas come from `FEATURES.ICECUBE86` and
`IceCube86.feature_map` in the pinned GraphNeT checkout. The raw production
units—metres, nanoseconds, photoelectrons, and so on—must still be confirmed
with the data owner; the code establishes transformations, not source
provenance. `pmt_area` is present in `data.x`, but `FourierEncoder` reads only
indices 0–5 when `n_features=6`. The baseline has `include_dynedge=false`, so
edges and the unused seventh feature do not enter a graph-convolution path.

Changing order or normalization invalidates the numerical meaning of existing
checkpoints even when tensor shapes still match.

## Fourier encoding

For an input scalar $u$ and an even embedding width $d$, GraphNeT constructs

$$
\begin{aligned}
\omega_j &= \exp\left[-j\,\frac{\log(n_{\mathrm{freq}})}{d/2}\right],
\qquad j=0,\ldots,d/2-1, \\
S_d(u) &= [\sin(u\omega_0),\ldots,\sin(u\omega_{d/2-1}),
           \cos(u\omega_0),\ldots,\cos(u\omega_{d/2-1})].
\end{aligned}
$$

With baseline `scaled_emb=true`, $S_d$ is additionally multiplied by
$d^{-1/2}$. IceMix supplies the following arguments:

$$
\begin{aligned}
E_{xyz} &= S_{128}(4096[x',y',z'])\quad\text{(flattened)}, \\
E_t &= S_{128}(4096t'), \\
E_q &= S_{128}(1024q'), \\
E_{rde} &= S_{128}(1024r'), \\
E_N &= S_{64}(\log_{10}N_{\mathrm{pulses}}).
\end{aligned}
$$

These pieces are concatenated and projected through linear–LayerNorm–GELU–linear
layers to the 384-dimensional hidden representation. `n_freq=10000` controls
the geometric frequency schedule; it is not literally a count of 10,000
separate frequencies.

## Pairwise spacetime-relative attention

For normalized pulse coordinates, the baseline `SpacetimeEncoder` computes

$$
\begin{aligned}
s_{ij} &= \|\mathbf{x}'_i-\mathbf{x}'_j\|^2
 - \left[18(t'_i-t'_j)\right]^2, \\
d_{ij} &= \operatorname{sign}(s_{ij})\sqrt{|s_{ij}|}, \\
b_{ij} &= W\,S_{32}\!\left(1024\,\operatorname{clip}(d_{ij},-4,4)\right).
\end{aligned}
$$

$b_{ij}$ becomes an additive relative-attention feature. All quantities in this
calculation are dimensionless after GraphNeT normalization. The value 18 is
algebraically consistent with converting the normalized time coordinate using
$0.3\ \mathrm{m/ns}$:

$$
18 = \frac{30{,}000\ \mathrm{ns}}{500\ \mathrm{m}}
     (0.3\ \mathrm{m/ns}).
$$

That observation does **not** make 18 a measured propagation speed in deep ice.
Its authorship, tuning history, and intended physical approximation are
**UNVERIFIED-CLUSTER**; ask the author before retuning it. With
`maha_encoder=true`, the minus sign becomes a plus and the signed square root is
replaced by an ordinary positive square root.

## Output and target layouts

The task head receives an unconstrained seven-vector. GraphNeT transforms it to

```text
[100 raw_x, 100 raw_y, 100 raw_z,
 raw_dx / ||raw_d||, raw_dy / ||raw_d||, raw_dz / ||raw_d||,
 abs(raw_kappa)]
```

IceMix and its metric callback therefore use prediction layout
`[pos_x, pos_y, pos_z, dir_x, dir_y, dir_z, kappa]` and target layout
`[pos_x, pos_y, pos_z, dir_x, dir_y, dir_z]`. Position predictions inherit the
target position unit; current code and logs call it metres. Direction components
are dimensionless, and `kappa` is a nonnegative dimensionless 3D-vMF
concentration.

## Joint objective

For event $i$, the configured objective is

$$
\begin{aligned}
L_{\mathrm{pos},i} &= \|\hat{\mathbf{r}}_i-\mathbf{r}_i\|_2, \\
L_{\mathrm{dir},i} &= -\log C_3(\kappa_i)
                      -\kappa_i\hat{\mathbf{d}}_i\!\cdot\!\mathbf{d}_i, \\
L_i &= \alpha L_{\mathrm{pos},i}+L_{\mathrm{dir},i}, \\
L_{\mathrm{batch}} &= \frac{1}{N}\sum_i L_i,
\qquad \alpha=0.026\ \text{by default}.
\end{aligned}
$$

No `oneweight` tensor is passed to this loss. The position term carries the
numerical scale of the position target, while the direction negative
log-likelihood is dimensionless; `alpha` is consequently a cross-scale modeling
choice, not a physical constant. Its original tuning criterion is
**UNVERIFIED-CLUSTER**. Changing position units without inversely rescaling
`alpha`, or comparing raw total losses from differently scaled targets, changes
the scientific objective.
