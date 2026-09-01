# v15 Q shared-row stride 144 static audit

## Symbols and fixed dimensions

| Symbol | Value | Meaning |
|---|---:|---|
| `G` | 16 | GQA query heads per KV head (`kGqaGroup`) / Q-tile rows |
| `D` | 128 | Live Q head dimension (`kHeadDim`) |
| `S12` | 136 | Frozen v12 Q-tile leading dimension |
| `S15` | 144 | v15 Q-tile leading dimension (`kQTileStride`) |
| `e` | 2 B | `__nv_bfloat16` element size |
| `R12` | 31,136 B | Frozen v12 AOT `SHARED` observation |
| `R15` | 31,392 B | Exact v15 AOT `SHARED` gate |

## Single-variable source audit

The patch targets the lifecycle-closed v12 source SHA-256
`535d90b856ed1062aa7b8a105eb2c5f236c450826e65496e646d0d5a27eb8aaf`
with fuzz `0`.  Its only functional delta is
`kQTileStride: kHeadDim + 8` (136) to `kHeadDim + 16` (144); the adjacent
comments describe that same change.

All eight existing `q_fragment_[0..7]` WMMA-A loads retain their base
addresses and use `kQTileStride`, so their effective leading dimension becomes
144 without changing fragment shape.  `144 % 8 == 0`, satisfying the source's
WMMA leading-dimension assertion.  The K/V staging declaration remains
`fp8_stage[kWarps][kTile][kTile]`, and its two WMMA loads retain leading
dimension `kTile`; no `kStageStride` symbol exists in v15.

## Shared-memory arithmetic and alignment proof

Only the Q tile grows:

`ΔQ = G × (S15 − S12) × e = 16 × (144 − 136) × 2 B = 256 B`.

The frozen Q tile is `16 × 136 × 2 B = 4,352 B`; v15 is
`16 × 144 × 2 B = 4,608 B`.  Both are multiples of the declaration's 32-B
alignment.  Consequently the subsequent unchanged 4,096-B K/V stage and all
following shared declarations retain their v12 offsets modulo 32, with no
extra layout padding.  The exact predicted resource gate is therefore:

`R15 = R12 + ΔQ = 31,136 B + 256 B = 31,392 B`.

The AOT gate remains authoritative: it requires `STACK=0`, `LOCAL=0`, and
`SHARED=31,392`; register count is recorded but not predeclared.  This proves
only static layout.  Compiler resources and directed runtime correctness
remain unverified until a later, separately approved submission.
