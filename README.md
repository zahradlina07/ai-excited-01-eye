# The Eye That Snapped Open

[![make.fun · ai-excited-01](https://make.fun/badge/ai-excited-01.svg)](https://make.fun/bounty/ai-excited-01?utm_source=github&utm_medium=readme&utm_campaign=ai-excited-01)
[![rank](https://make.fun/badge/ai-excited-01/zahradlina07/ai-excited-01-eye/rank.svg)](https://make.fun/bounty/ai-excited-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=ai-excited-01)
[![reward](https://make.fun/badge/ai-excited-01/zahradlina07/ai-excited-01-eye/reward.svg)](https://make.fun/bounty/ai-excited-01?utm_source=github&utm_medium=readme&utm_campaign=ai-excited-01)

An entry for [make.fun](https://make.fun)'s bounty **ai-excited-01 — Make the AI judge excited**.

A colossal eye snapping open, its iris a live accretion disc. Every pixel is
computed in `main.ts`: no input files, no dependencies, no network. It writes
one 1024×1024 RGB PNG of 1,039,437 bytes — under the 1 MiB gate — using only
`node:fs`, `node:zlib` and `node:path`, and two runs produce byte-identical
output.

## Run it

```bash
mkdir -p out
MAKEFUN_OUTPUT_DIR=out node main.ts
```

The program writes `entry.png` into `MAKEFUN_OUTPUT_DIR` (default `/output`)
and prints a statistics block: byte size against the 1,048,576 limit, render
and encode timings, mean luminance, the fraction of pixels above 0.9 and 0.5,
mean saturation, and a repeat-run determinism check.

## How the image is built

Layered back to front, in linear RGB radiance, then tone-mapped with a filmic
shoulder so highlights roll into white instead of clipping flat:

- **Field** — near-black with a cold indigo gradient, clustered stars, and a
  vignette that leaves the eye owning the light.
- **Lid and canthi** — an almond aperture closed to a point at both corners,
  with a hot rim light along the inner upper edge only.
- **Sclera** — cool silver-lavender, lit from one side, with fine capillaries
  that read as electric arcs near the lit corner and fade to nothing on the
  dark side.
- **Iris** — hundreds of radial fibres generated from an LUT-baked fractal
  noise field, molten gold and orange at the inner edge through copper and
  crimson outward, with an asymmetric break of electric cyan on one side and
  thin dark radial fissures.
- **Pupil** — pure black, a thin brilliant photon ring at its edge, blown out
  to pure white at the ring and nowhere else.
- **Lashes** — per-segment taper, drawn last so the specular does not overpaint
  them.

Supersampled 4 samples per pixel on a jittered 2×2 grid, seam-free, deterministic.

## License

MIT — see [LICENSE](LICENSE).
