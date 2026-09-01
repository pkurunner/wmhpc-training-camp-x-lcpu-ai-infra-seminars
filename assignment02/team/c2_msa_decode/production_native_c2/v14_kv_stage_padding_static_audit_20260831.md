# v14 K/V stage-padding static audit

## Symbols and fixed dimensions

| Symbol | Value | Meaning |
|---|---:|---|
| `W` | 8 | CTA warps (`kWarps`) |
| `T` | 16 | WMMA tile rows and live BF16 columns (`kTile`) |
| `S12` | 16 | v12 K/V stage leading dimension |
| `S14` | 24 | v14 K/V stage leading dimension (`kStageStride`) |
| `e` | 2 B | `__nv_bfloat16` element size |
| `R12` | 31,136 B | Frozen v12 AOT `SHARED` observation used as v14 baseline |

## Single-variable source audit

The patch is against the frozen v12 source only and changes only:

1. the stage-row leading dimension from `kTile` to `kStageStride = kTile + 8`;
2. the stage allocation's final dimension; and
3. the existing K and V `wmma::load_matrix_sync` leading dimensions.

Both stores retain `fp8_stage[warp][token][dim]` with `token, dim` in
`[0, 15]`; no index reaches the eight padding columns.  The two affected
WMMA loads consume the same base address and 16x16 fragment shape as v12,
with an explicitly `static_assert`ed multiple-of-eight leading dimension.

## Shared-memory arithmetic and alignment proof

The stage size change is

`Δstage = W × T × (S14 − S12) × e = 8 × 16 × 8 × 2 B = 2,048 B`.

`q_tile` occupies `16 × 136 × 2 B = 4,352 B`, a multiple of its 32-B
alignment.  `fp8_stage` has explicit 32-B alignment and both its v12 size
(`8 × 16 × 16 × 2 B = 4,096 B`) and v14 size
(`8 × 16 × 24 × 2 B = 6,144 B`) are multiples of 32 B.  Therefore the stage
start alignment and every following declared shared object's alignment are
unchanged; no new static-layout padding is introduced after this object.

Under the frozen v12 resource baseline, the exact expected v14 gate is

`R14 = R12 + Δstage = 31,136 B + 2,048 B = 33,184 B`.

The AOT script still treats `cuobjdump` as authoritative: `STACK=0`,
`LOCAL=0`, and `SHARED=33,184` must all hold.  This static proof does not
substitute for compiler resource output or directed runtime correctness.
