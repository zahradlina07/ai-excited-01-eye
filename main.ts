/**
 * main.ts — "COLOSSAL EYE": a vast eye filling the frame, the instant it snaps open,
 * its iris a live accretion disc.
 *
 * Node 22+, standard library only. Every pixel computed. Deterministic.
 * Writes <MAKEFUN_OUTPUT_DIR>/entry.png (1024x1024, 8-bit RGB, colour type 2).
 */

import * as fs from "node:fs";
import * as zlib from "node:zlib";
import * as path from "node:path";
import * as crypto from "node:crypto";

// ---------------------------------------------------------------- constants
const W = 1024;
const H = 1024;
const CX = 512.0;
const CY = 508.0;

const A = 512.0; // eye half width at the canthi
const B = 432.0; // vertical scale of the almond

const TILT = 0.034; // axis tilt: outer corner low, inner corner high
const CT = Math.cos(TILT);
const ST = Math.sin(TILT);

const R_PUPIL = 0.280 * A; // 143.4
const R_IRIS = 0.520 * A; // 266.2
const IRIS_SPAN = R_IRIS - R_PUPIL;

const TAU = Math.PI * 2;

// ---------------------------------------------------------------- utilities
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rng = mulberry32(0x5eed1234);
const clamp = (x: number, a: number, b: number) => (x < a ? a : x > b ? b : x);
const mix = (a: number, b: number, t: number) => a + (b - a) * t;
function smoothstep(e0: number, e1: number, x: number): number {
  let t = (x - e0) / (e1 - e0);
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return t * t * (3 - 2 * t);
}
const sq = (x: number) => x * x;

function h2i(x: number, y: number, s: number): number {
  let h = Math.imul(x | 0, 0x27d4eb2d) ^ Math.imul(y | 0, 0x165667b1) ^ Math.imul(s | 0, 0x9e3779b1);
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function vnoise(x: number, y: number, s: number): number {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const a = h2i(ix, iy, s), b = h2i(ix + 1, iy, s);
  const c = h2i(ix, iy + 1, s), d = h2i(ix + 1, iy + 1, s);
  return mix(mix(a, b, ux), mix(c, d, ux), uy);
}

function fbm(x: number, y: number, s: number, oct: number): number {
  let v = 0, amp = 0.5, f = 1;
  for (let i = 0; i < oct; i++) {
    v += amp * vnoise(x * f, y * f, s + i * 17);
    amp *= 0.5;
    f *= 2.07;
  }
  return v;
}

// pre-baked noise textures (cheap lookups in the inner loop)
const NT = 256;
const nTex = new Float32Array(NT * NT); // large-scale organic mottle
const nTex2 = new Float32Array(NT * NT); // fine grain
for (let y = 0; y < NT; y++) {
  for (let x = 0; x < NT; x++) {
    const u = (x / NT) * TAU, v = (y / NT) * TAU;
    let s = 0;
    s += Math.sin(u * 2 + Math.cos(v * 3) * 1.7) * 0.5 + Math.sin(v * 2 + Math.cos(u * 1) * 1.3) * 0.5;
    s += Math.sin(u * 5 + v * 3) * 0.25 + Math.cos(u * 7 - v * 4) * 0.18;
    nTex[y * NT + x] = clamp(s * 0.5 + 0.5, 0, 1);
    nTex2[y * NT + x] = fbm(x * 0.11, y * 0.11, 90, 3);
  }
}

function texBil(t: Float32Array, x: number, y: number): number {
  let fx = (x * (NT / 1024)) % NT;
  let fy = (y * (NT / 1024)) % NT;
  if (fx < 0) fx += NT;
  if (fy < 0) fy += NT;
  const ix = fx | 0, iy = fy | 0;
  const tx = fx - ix, ty = fy - iy;
  const x0 = ix & (NT - 1), x1 = (ix + 1) & (NT - 1);
  const i0 = (iy & (NT - 1)) * NT, i1 = ((iy + 1) & (NT - 1)) * NT;
  const a = t[i0 + x0], b = t[i0 + x1], c = t[i1 + x0], d = t[i1 + x1];
  return mix(mix(a, b, tx), mix(c, d, tx), ty);
}

function bilArr(arr: Float32Array, S: number, x: number, y: number): number {
  let fx = x - 0.5, fy = y - 0.5;
  if (fx < 0) fx = 0;
  if (fy < 0) fy = 0;
  if (fx > S - 1.001) fx = S - 1.001;
  if (fy > S - 1.001) fy = S - 1.001;
  const ix = fx | 0, iy = fy | 0;
  const tx = fx - ix, ty = fy - iy;
  const i0 = iy * S + ix, i1 = i0 + S;
  const a = arr[i0], b = arr[i0 + 1], c = arr[i1], d = arr[i1 + 1];
  return mix(mix(a, b, tx), mix(c, d, tx), ty);
}

// ---------------------------------------------------------------- lid shape
const LUT = 4096;
const upTab = new Float32Array(LUT + 1);
const loTab = new Float32Array(LUT + 1);

function shapeUp(t: number): number {
  // heavily dominant upper lid: full over the globe, long sharp sweep to the canthi
  const u = t < 0 ? Math.abs(t / 0.995) : Math.abs(t / 1.0);
  if (u >= 1) return 0;
  const k = t < 0 ? 1.70 : 1.90;
  return Math.pow(1 - Math.pow(u, k), 0.66);
}
function shapeLo(t: number): number {
  const u = t < 0 ? Math.abs(t / 0.985) : Math.abs(t / 0.997);
  if (u >= 1) return 0;
  return Math.pow(1 - Math.pow(u, 2.15), 0.88);
}

for (let i = 0; i <= LUT; i++) {
  const t = (i / LUT) * 2 - 1;
  const su = shapeUp(t), sl = shapeLo(t);
  // irregular lid edge: low-frequency wobble, faded at the canthi so they stay needle sharp
  const wob =
    5.2 * (vnoise(t * 4.0 + 3.1, 11.4, 5) - 0.5) +
    2.4 * (vnoise(t * 11.0 + 8.2, 21.9, 9) - 0.5) +
    1.0 * (vnoise(t * 27.0 + 1.7, 4.4, 13) - 0.5);
  const wob2 =
    2.4 * (vnoise(t * 5.0 + 21.1, 2.7, 31) - 0.5) +
    1.0 * (vnoise(t * 17.0 + 5.5, 17.1, 37) - 0.5);
  upTab[i] = -B * 0.955 * su + wob * Math.pow(su, 1.35);
  loTab[i] = B * 0.600 * sl + wob2 * Math.pow(sl, 1.30);
}

function upAt(t: number): number {
  const f = (t + 1) * 0.5 * LUT;
  if (f <= 0) return upTab[0];
  if (f >= LUT) return upTab[LUT];
  const i = f | 0;
  return mix(upTab[i], upTab[i + 1], f - i);
}
function loAt(t: number): number {
  const f = (t + 1) * 0.5 * LUT;
  if (f <= 0) return loTab[0];
  if (f >= LUT) return loTab[LUT];
  const i = f | 0;
  return mix(loTab[i], loTab[i + 1], f - i);
}

// ---------------------------------------------------------------- iris fibres
const FIB = 236;
const dAng = TAU / FIB;
const fibW = new Float32Array(FIB); // half width in px
const fibB = new Float32Array(FIB); // luminance
const fibOff = new Float32Array(FIB); // angular offset (cells)
const fibTone = new Float32Array(FIB); // 0 = crimson, 1 = pale gold
const fibR0 = new Float32Array(FIB);
const fibR1 = new Float32Array(FIB);
const fibFrq = new Float32Array(FIB);
const fibPh = new Float32Array(FIB);
const fibDk = new Float32Array(FIB); // 1 = fissure (dark strand)
const fibGap = new Float32Array(FIB); // 1 = crypt (no fibre at all)

for (let i = 0; i < FIB; i++) {
  const w = rng();
  // half-widths in px at the *inner* radius: wide enough to survive 4x AA
  fibW[i] = w < 0.18 ? 2.9 + rng() * 1.7 : w < 0.58 ? 1.5 + rng() * 1.3 : 0.85 + rng() * 0.75;
  fibB[i] = 0.34 + Math.pow(rng(), 1.10) * 1.35;
  fibOff[i] = (rng() - 0.5) * 0.95;
  fibTone[i] = rng();
  fibR0[i] = R_PUPIL + 1.0 + Math.pow(rng(), 2.2) * 0.18 * IRIS_SPAN;
  fibR1[i] = R_IRIS - 1.2 - Math.pow(rng(), 1.9) * 0.20 * IRIS_SPAN;
  fibFrq[i] = 0.028 + rng() * 0.115;
  fibPh[i] = rng() * TAU;
  fibDk[i] = rng() < 0.10 ? 1 : 0;
  fibGap[i] = rng() < 0.05 ? 1 : 0;
}

// ---------------------------------------------------------------- masks
const LX = new Float32Array(W * H); // lash cores (world space)
const SX = new Float32Array(W * H); // capillary cores (world space)
const SH = 256;
const LSH = new Float32Array(SH * SH); // lash shadow (low res, blurred)

function toEyeX(dx: number, dy: number) {
  return CT * dx + ST * dy;
}
function toEyeY(dx: number, dy: number) {
  return -ST * dx + CT * dy;
}
function ewx(ex: number, ey: number): number {
  return CX + CT * ex - ST * ey;
}
function ewy(ex: number, ey: number): number {
  return CY + ST * ex + CT * ey;
}

/** paint a disc into a mask; maxMode keeps the strongest value */
function stamp(mask: Float32Array, S: number, x: number, y: number, r: number, amp: number, maxMode: boolean) {
  const r2 = r * r;
  const x0 = Math.max(0, Math.floor(x - r - 1)), x1 = Math.min(S - 1, Math.ceil(x + r + 1));
  const y0 = Math.max(0, Math.floor(y - r - 1)), y1 = Math.min(S - 1, Math.ceil(y + r + 1));
  for (let py = y0; py <= y1; py++) {
    const ddy = py + 0.5 - y;
    for (let px = x0; px <= x1; px++) {
      const ddx = px + 0.5 - x;
      const d2 = ddx * ddx + ddy * ddy;
      if (d2 > (r + 1) * (r + 1)) continue;
      const v = amp * Math.exp(-d2 / (r2 * 0.55));
      const k = py * S + px;
      if (maxMode) {
        if (v > mask[k]) mask[k] = v;
      } else {
        mask[k] += v;
      }
    }
  }
}

/** stroke a polyline whose radius tapers from wRoot to wTip */
function stroke(mask: Float32Array, S: number, pts: Array<[number, number]>, wRoot: number, wTip: number) {
  const total = pts.length - 1;
  for (let i = 0; i < total; i++) {
    const [x0, y0] = pts[i], [x1, y1] = pts[i + 1];
    const len = Math.hypot(x1 - x0, y1 - y0);
    const n = Math.max(1, Math.ceil(len / 0.55));
    for (let s = 0; s < n; s++) {
      const f = (i + s / n) / total;
      const px = x0 + (x1 - x0) * (s / n);
      const py = y0 + (y1 - y0) * (s / n);
      const w = mix(wRoot, wTip, f);
      if (w < 0.12) continue;
      stamp(mask, S, px, py, w, 1.0, true);
    }
  }
}

function boxBlur(src: Float32Array, S: number, r: number, passes: number) {
  let a = src, b = new Float32Array(S * S);
  for (let p = 0; p < passes; p++) {
    for (let y = 0; y < S; y++) {
      const row = y * S;
      for (let x = 0; x < S; x++) {
        let s = 0, n = 0;
        for (let k = -r; k <= r; k++) {
          const xx = x + k;
          if (xx < 0 || xx >= S) continue;
          s += a[row + xx];
          n++;
        }
        b[row + x] = s / n;
      }
    }
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        let s = 0, n = 0;
        for (let k = -r; k <= r; k++) {
          const yy = y + k;
          if (yy < 0 || yy >= S) continue;
          s += b[yy * S + x];
          n++;
        }
        a[y * S + x] = s / n;
      }
    }
  }
  return a;
}

// ---- lashes: sparse, long, thin, rooted on the upper-left lid margin,
//      hanging down-right at a low angle so they cut across the iris.
{
  const roots = [-0.72, -0.58, -0.45, -0.33, -0.21, -0.08];
  for (let i = 0; i < roots.length; i++) {
    const t = roots[i] + (rng() - 0.5) * 0.028;
    const ex0 = t * A;
    const ey0 = upAt(t) + 1.0;
    const len = 520 + rng() * 240;
    const drop = (0.62 + rng() * 0.30) * len; // steep enough to cut the iris
    const wid = 2.1 + rng() * 1.6;
    const pts: Array<[number, number]> = [];
    const N = 26;
    let cx2 = ex0, cy2 = ey0;
    pts.push([ewx(cx2, cy2), ewy(cx2, cy2)]);
    for (let k = 1; k <= N; k++) {
      const uu = k / N;
      // start along the lid, then swing down and right
      const ang = 0.10 + Math.pow(uu, 1.45) * 1.02 + Math.sin(uu * 2.6 + i) * 0.13;
      const step = len / N;
      cx2 += Math.cos(ang) * step;
      cy2 += Math.sin(ang) * (drop / len) * step * 1.5;
      pts.push([ewx(cx2, cy2), ewy(cx2, cy2)]);
    }
    // stamp segment by segment with a taper: thick at the root, drawn to a
    // point at the tip, so the strand reads as hair and not a ruled scratch.
    for (let s = 0; s + 1 < pts.length; s++) {
      const u = s / (pts.length - 1);
      const wSeg = wid * (1.0 - 0.82 * Math.pow(u, 1.35));
      stroke(LX, W, [pts[s], pts[s + 1]], Math.max(0.55, wSeg), Math.max(0.45, wSeg * 0.90));
    }
    // soft cast shadow, displaced away from the key light (down + right)
    const sp: Array<[number, number]> = [];
    for (let s = 0; s < pts.length; s++) {
      const u = s / (pts.length - 1);
      sp.push([(pts[s][0] + 10 + u * 22) * (SH / W), (pts[s][1] + 14 + u * 18) * (SH / W)]);
    }
    stroke(LSH, SH, sp, wid * 1.6, 0.55);
  }
  boxBlur(LSH, SH, 3, 2);
}

// ---- capillaries: a handful of fine strands, only near the corners
{
  const mk = (side: number) => {
    const nb = 5 + Math.floor(rng() * 2);
    for (let i = 0; i < nb; i++) {
      const ex0 = side * (0.66 + rng() * 0.26) * A;
      const ey0 = (rng() - 0.5) * 0.30 * B + 8;
      const len = 80 + rng() * 120;
      const dir = -side;
      const pts: Array<[number, number]> = [];
      let ex = ex0, ey = ey0;
      const nSeg = 20;
      for (let s = 0; s <= nSeg; s++) {
        pts.push([ewx(ex, ey), ewy(ex, ey)]);
        const u = s / nSeg;
        ex += dir * (len / nSeg) * (0.95 + 0.22 * Math.sin(u * 4.2 + i * 2.0));
        ey += (len / nSeg) * 0.42 * Math.sin(u * 2.4 + i * 2.7) * (1 - u * 0.4);
      }
      stroke(SX, W, pts, 0.75 + rng() * 0.5, 0.2);
      if (rng() < 0.6) {
        const k0 = 4 + Math.floor(rng() * 7);
        const br: Array<[number, number]> = [];
        let bx = pts[k0][0], by = pts[k0][1];
        for (let s = 0; s <= 9; s++) {
          br.push([bx, by]);
          bx += dir * (len / 30);
          by += 4.0 + (rng() - 0.5) * 5;
        }
        stroke(SX, W, br, 0.55, 0.18);
      }
    }
  };
  mk(-1);
  mk(1);
}

// ---------------------------------------------------------------- iris palette
const STOPS: Array<[number, number, number, number]> = [
  [0.00, 3.10, 1.78, 0.52],
  [0.06, 2.30, 1.12, 0.235],
  [0.18, 1.78, 0.70, 0.105],
  [0.36, 1.34, 0.375, 0.062],
  [0.55, 0.98, 0.190, 0.050],
  [0.74, 0.54, 0.082, 0.036],
  [0.88, 0.19, 0.031, 0.024],
  [1.00, 0.028, 0.010, 0.011],
];

const aria = new Float32Array(3);
function palette(rn: number, out: Float32Array) {
  let i = 0;
  while (i < STOPS.length - 2 && rn > STOPS[i + 1][0]) i++;
  const a = STOPS[i], b = STOPS[i + 1];
  const t = clamp((rn - a[0]) / (b[0] - a[0]), 0, 1);
  const ts = t * t * (3 - 2 * t);
  out[0] = mix(a[1], b[1], ts);
  out[1] = mix(a[2], b[2], ts);
  out[2] = mix(a[3], b[3], ts);
}

const CA_P0 = 0.62; // cyan break: screen angle +x/+y => ~4 o'clock
const col = new Float32Array(3);
let lastPupil = 0; // pupil coverage of the last sample, for bloom/sheen gating

// ---------------------------------------------------------------- shading
function samplePixel(x: number, y: number, out: Float32Array, gr: number) {
  const dx = x - CX, dy = y - CY;
  const ex = toEyeX(dx, dy), ey = toEyeY(dx, dy);

  const tRaw = ex / A;
  const tIn = clamp(tRaw, -1, 1);
  const up = upAt(tIn);
  const lo = loAt(tIn);
  const du = ey - up; // >0 => below the upper lid margin (inside)
  const dl = lo - ey; // >0 => above the lower lid margin (inside)
  const inset = Math.min(du, dl);
  const inX = smoothstep(1.006, 0.994, Math.abs(tRaw));
  const inside = smoothstep(-0.6, 0.6, inset) * inX;

  // ---- field: near-black with a faint cold gradient
  const nx = dx / (W * 0.60), ny = dy / (H * 0.62);
  const q = Math.sqrt(nx * nx + ny * ny * 0.86);
  const vig = 1.0 / (1.0 + Math.pow(q * 1.30, 3.4));
  const n1 = texBil(nTex, x * 0.30, y * 0.30);
  const n2 = texBil(nTex2, x * 1.05, y * 1.05);
  const n3 = texBil(nTex2, x * 0.16 + 61, y * 0.16 + 17);

  let sr = 0.0105, sg = 0.0122, sb = 0.0185;
  const fldMod = vig * (0.55 + 0.72 * n1) * (0.78 + 0.44 * n2);
  sr *= fldMod; sg *= fldMod; sb *= fldMod * 1.06;

  // ---- matte skin around the lids
  {
    const apartUp = -du, apartLo = -dl;
    const sideLight = 0.22 + 0.98 * smoothstep(0.95, -1.0, tRaw);
    // broad matte brow mass and a lit ridge hugging the lid margin
    const brow = Math.exp(-sq((apartUp - 120.0) / 105.0)) * 0.052 * vig * (0.40 + 0.95 * n3);
    const browHi = Math.exp(-sq((apartUp - 88.0) / 44.0)) * 0.020 * vig * sideLight * (0.6 + 0.7 * n1);
    const ridgeU = Math.exp(-sq((apartUp - 9.0) / 7.4)) * 0.105 * sideLight * vig;
    const ridgeL = Math.exp(-sq((apartLo - 7.0) / 6.6)) * 0.040 * sideLight * vig;
    const crease = Math.exp(-sq((apartUp - 27.0) / 12.0)) * 0.78;
    const socket = Math.exp(-sq((apartUp - 190.0) / 150.0)) * 0.34;
    sr += brow + browHi + ridgeU * 1.30 + ridgeL * 1.05;
    sg += brow * 0.64 + browHi * 0.60 + ridgeU * 0.74 + ridgeL * 0.63;
    sb += brow * 0.44 + browHi * 0.40 + ridgeU * 0.44 + ridgeL * 0.52;
    const dk = clamp(crease + socket, 0, 1);
    sr *= 1 - dk; sg *= 1 - dk; sb *= 1 - dk * 0.88;
    // the sharp dark lid edge
    const edgeU = Math.exp(-sq(apartUp / 2.2));
    const edgeL = Math.exp(-sq(apartLo / 1.6));
    sr *= 1 - 0.96 * edgeU - 0.86 * edgeL;
    sg *= 1 - 0.96 * edgeU - 0.86 * edgeL;
    sb *= 1 - 0.93 * edgeU - 0.80 * edgeL;
    sr *= 1.5; sg *= 1.16; sb *= 1.02;
    const sk = 1 + (n3 - 0.5) * 1.05 + (n2 - 0.5) * 0.30;
    sr *= sk; sg *= sk; sb *= sk;
  }

  out[0] = sr; out[1] = sg; out[2] = sb;
  if (inside <= 0.0015) return out;

  // ---- inside the aperture
  const r = Math.hypot(ex, ey);
  const th = Math.atan2(ey, ex);
  const ir = clamp((r - R_PUPIL) / IRIS_SPAN, 0, 1.3);
  const corner = smoothstep(0.28, 1.0, Math.abs(tRaw));

  // ============ sclera: deep desaturated blue-grey, lit by the iris
  const lidShadow = 1 - 0.78 * Math.exp(-Math.max(0, du) / 22.0);
  const glow = 0.60 / (1.0 + Math.pow(Math.max(0, r - R_IRIS * 0.94) / 88.0, 2.4));
  let sc_r = 0.0130 * lidShadow + glow * 1.05;
  let sc_g = 0.0220 * lidShadow + glow * 0.755;
  let sc_b = 0.0455 * lidShadow + glow * 0.565;
  // cold shadow falloff toward the outer corner, so the warm iris reads hotter
  const coldFar = smoothstep(-0.15, -1.0, tRaw);
  sc_r *= 1 - 0.45 * coldFar;
  sc_g *= 1 - 0.32 * coldFar;
  sc_b *= 1 - 0.05 * coldFar;
  sc_r *= 1 - 0.66 * corner;
  sc_g *= 1 - 0.58 * corner;
  sc_b *= 1 - 0.38 * corner;
  // radial tension streaks: the eye is about to snap fully open
  const streakMod = 1 + 0.085 * Math.sin(th * 47.0 + 2.4 * n1) + 0.045 * Math.sin(th * 91.0 + 5.0 * n3);
  sc_r *= streakMod; sc_g *= streakMod; sc_b *= streakMod;
  // tissue micro-structure: mottled, slightly cloudy, never a clean gradient fill
  const sct = 1 + (n3 - 0.5) * 1.05 + (n1 - 0.5) * 0.40 + (n2 - 0.5) * 0.30;
  sc_r *= sct; sc_g *= sct; sc_b *= sct;
  // a cool cast in the shaded upper sclera, warm bounce below
  const tbl = (ey / B);
  sc_r *= 1 - 0.10 * smoothstep(0.1, -0.9, tbl);
  sc_b *= 1 + 0.14 * smoothstep(0.1, -0.9, tbl);
  // lid contact occlusion: a hard dark line where the lid actually meets the globe
  const contact = Math.exp(-Math.max(0, du) / 5.0) * 0.85 + Math.exp(-Math.max(0, dl) / 4.0) * 0.55;
  sc_r *= 1 - 0.55 * contact; sc_g *= 1 - 0.55 * contact; sc_b *= 1 - 0.50 * contact;

  // blood capillaries near the corners only
  const capM = bilArr(SX, W, x, y);
  if (capM > 0.002) {
    const cw = Math.min(1.0, capM * (0.18 + 0.85 * corner));
    sc_r += cw * 1.15;
    sc_g += cw * 0.030;
    sc_b += cw * 0.055;
    sc_r *= 1 + cw * 0.55;
  }
  // caustic of the iris colour spilling onto the sclera below
  const below = smoothstep(-0.05, 0.75, ey / 150.0);
  const caus = Math.exp(-sq((r - R_IRIS - 10.0) / 58.0)) * below;
  sc_r += caus * 0.62; sc_g += caus * 0.225; sc_b += caus * 0.070;

  // ============ iris
  if (r > R_PUPIL - 2 && r < R_IRIS + 4) {
    palette(ir, aria);
    const pr = aria[0], pg = aria[1], pb = aria[2];

    // ---- strand field: take the NEAREST fibre, not a sum. Summing many wide
    // Gaussians saturates to a flat wash; nearest-fibre distance keeps honest
    // dark gaps between countable filaments.
    const k = (th / dAng + FIB * 16) % FIB;
    // `twist` is a phase offset in fibre units; it can be several tens of units
    // at the inner edge, so the neighbour search must be anchored on it. Searching
    // around kk instead silently misses every fibre except at the periphery.
    const twist = (0.70 * Math.pow(clamp(1 - ir, 0, 1), 1.6)) / dAng;
    const kk = Math.floor(k - twist) + 1;
    const motion = Math.pow(clamp(1 - ir, 0, 1), 3.0);
    let bestArc = 1e9, bi = kk;
    for (let o = -2; o <= 2; o++) {
      let i = kk + o;
      if (i < 0) i += FIB; else if (i >= FIB) i -= FIB;
      const wob = fibOff[i] + 0.40 * Math.sin(r * 0.042 + fibPh[i]) + 0.13 * Math.sin(r * 0.13 + fibPh[i] * 2.3);
      let dk2 = k - twist - (i + wob);
      dk2 -= Math.round(dk2 / FIB) * FIB;
      const arc = Math.abs(dk2 * dAng) * r;
      if (arc < bestArc) { bestArc = arc; bi = i; }
    }
    // the winning fibre's own profile
    const i0 = bi;
    // Motion blur toward the inner edge, but only enough to smear the fibre
    // heads: past ~2x width the strands merge and the inner iris goes blank.
    const w0 = fibW[i0] * (1.0 + motion * 1.15);
    // coverage: 1 at the strand core, falling to 0 in the gap
    let core = Math.exp(-sq(bestArc / w0) * 2.2);
    // radial extent, soft at head and tail
    const e0 = smoothstep(fibR0[i0] - 10, fibR0[i0] + 6, r);
    const e1 = smoothstep(fibR1[i0] + 9, fibR1[i0] - 8, r);
    core *= (0.14 + 0.86 * e0) * (0.18 + 0.82 * e1);
    if (fibGap[i0] > 0 && r > R_PUPIL + 0.22 * IRIS_SPAN) core *= 0.12;
    const tone = fibTone[i0];
    const lumF = fibB[i0] * (0.62 + 0.38 * Math.sin(r * fibFrq[i0] * 6.0 + fibPh[i0] * 2.0));
    const darkF = fibDk[i0] > 0 ? core : 0;
    void bestArc;
    // plus a low-frequency luminous ground so the gaps are dark, not black
    core = clamp(core, 0, 1);

    // strand colour at this radius: gold -> copper -> crimson by tone
    const fR = pr * (0.78 + tone * 0.50) * (0.55 + 0.85 * lumF);
    const fG = pg * (0.56 + (1 - tone) * 0.90) * (0.55 + 0.85 * lumF);
    const fB = pb * (0.42 + tone * 0.66) * (0.55 + 0.85 * lumF);

    // gap colour: much darker, cool-shifted
    const gapK = 0.070 + 0.045 * Math.sin(ir * 13.0 + th * 2.0);
    const gR = pr * gapK, gG = pg * gapK * 0.78, gB = pb * gapK * 0.70;

    // smooth only over ~1 px so filament edges stay crisp
    const solidS = core * core * (3 - 2 * core);
    let ir_r = mix(gR, fR, solidS);
    let ir_g = mix(gG, fG, solidS);
    let ir_b = mix(gB, fB, solidS);

    // ---- crypts: angular gaps in the stroma
    const cryptA = 0.5 + 0.5 * Math.sin(th * 8.0 + 1.3 + 1.6 * Math.sin(ir * 5.0));
    const cryptB = 0.5 + 0.5 * Math.sin(th * 19.0 + 4.1);
    const crypt = Math.pow(cryptA, 8.0) * Math.pow(cryptB, 1.5) *
      smoothstep(0.30, 0.58, ir) * (1 - smoothstep(0.94, 1.0, ir));

    // ---- spiral dust lanes winding into the pupil
    const lr = Math.log(clamp(r / R_PUPIL, 1.0001, 3.2));
    const sp1 = 0.5 + 0.5 * Math.sin(2.0 * th + 7.2 * lr - 1.15);
    const sp2 = 0.5 + 0.5 * Math.sin(4.0 * th + 13.0 * lr + 2.30);
    const lane = Math.pow(sp1, 3.0) * 0.80 + Math.pow(sp2, 5.0) * 0.35;
    const dlanes = Math.pow(1 - sp1, 3.0) * 0.55 * (1 - ir * 0.5);
    const mFade = 1.0 - 0.45 * motion;

    // bright dust spiralling in
    const laneAdd = lane * mFade * (0.30 + 0.90 * solidS);
    ir_r += pr * laneAdd * 0.55;
    ir_g += pg * laneAdd * 0.55;
    ir_b += pb * laneAdd * 0.55;

    // contraction furrows across the outer iris
    let fur = 0;
    for (let fi = 0; fi < 5; fi++) {
      const rr = R_PUPIL + (0.44 + fi * 0.112) * IRIS_SPAN + 4.5 * Math.sin(th * 2.6 + fi * 1.9);
      fur += Math.exp(-sq((r - rr) / 2.1)) * (0.34 - fi * 0.042);
    }
    // collarette: raised, knotty ridge at ~0.30 span
    const colR = R_PUPIL + 0.30 * IRIS_SPAN + 2.8 * Math.sin(th * 6.0);
    const colBand = Math.exp(-sq((r - colR) / 7.0)) * (0.50 + 0.50 * Math.sin(th * 11.0 + 2.0));
    ir_r *= 1 + colBand * 0.55; ir_g *= 1 + colBand * 0.42; ir_b *= 1 + colBand * 0.30;

    const dkAll = clamp(dlanes + crypt * 0.85 + fur, 0, 0.96);
    ir_r *= 1 - dkAll; ir_g *= 1 - dkAll; ir_b *= 1 - dkAll * 1.02;
    ir_r *= 1 - darkF * 0.75; ir_g *= 1 - darkF * 0.78; ir_b *= 1 - darkF * 0.80;

    // ---- cyan break at ~4 o'clock: a radial wedge of cold plasma through the fibres
    let da = th - CA_P0;
    da -= Math.round(da / TAU) * TAU;
    const wing = Math.exp(-sq(da / 0.44)) + 0.42 * Math.exp(-sq((da - 0.70) / 0.26))
      + 0.26 * Math.exp(-sq((da + 0.50) / 0.20));
    const rad = smoothstep(0.10, 0.30, ir) * smoothstep(0.995, 0.66, ir);
    const cw = wing * rad;
    if (cw > 0.002) {
      const chot = clamp(1 - Math.abs(ir - 0.46) * 2.0, 0, 1);
      const cr = 0.020 + chot * 0.10, cg = 0.66 + chot * 0.70, cb = 0.86 + chot * 0.46;
      // the fibres keep running through the cold patch: modulate by strand coverage
      const fibreMod = 0.42 + 0.72 * solidS + 0.20 * Math.sin(r * 0.09 + th * 9.0) * solidS;
      const km = clamp(cw * (0.85 + 0.45 * solidS), 0, 1);
      ir_r = mix(ir_r, Math.max(ir_r * 0.12, cr * fibreMod), km * 0.97);
      ir_g = mix(ir_g, Math.max(ir_g * 0.28, cg * fibreMod), km * 0.97);
      ir_b = mix(ir_b, Math.max(ir_b * 0.40, cb * fibreMod), km * 0.97);
    }
    // faint cold echo opposite, so the cyan reads as pigment, not a sticker
    let db = th - (CA_P0 + Math.PI);
    db -= Math.round(db / TAU) * TAU;
    const echo = Math.exp(-sq(db / 0.55)) * smoothstep(0.30, 0.55, ir) * smoothstep(0.98, 0.80, ir) * 0.10;
    ir_r *= 1 - echo * 0.55; ir_g *= 1 - echo * 0.20; ir_b *= 1 + echo * 0.10;

    // ---- limbal ring: hard, dark, irregular, and lit from the upper left
    const limSide = 0.45 + 0.75 * smoothstep(-0.9, 1.1, (th - 0.55) / TAU);
    const limbW = 0.045 + 0.030 * Math.sin(th * 3.0 + 0.8) + 0.016 * Math.sin(th * 11.0);
    const limb = smoothstep(1.0 - limbW, 1.0, ir);
    ir_r *= 1 - 0.95 * limb; ir_g *= 1 - 0.96 * limb; ir_b *= 1 - 0.94 * limb;
    // a thin crescent of limbal light on the shadow side, so the globe reads round
    const limbHi = Math.exp(-sq((ir - 0.955) / 0.030)) * smoothstep(0.2, 1.0, limSide) * 0.30;
    ir_r += limbHi * 0.55; ir_g += limbHi * 0.42; ir_b += limbHi * 0.34;

    // ---- the iris catches fire at the pupil edge.
    // These are GAINS on the fibre colour, not additive white: adding radiance
    // here saturates the tone curve and flattens the strands into a smooth glow.
    const fire = Math.exp(-sq((r - R_PUPIL - 4.0) / 15.0));
    ir_r *= 1 + fire * 1.55; ir_g *= 1 + fire * 0.62; ir_b *= 1 + fire * 0.22;

    // pupil edge: a hot but *uneven* rim — thickness and brightness both vary
    // with angle, so it reads as light caught on tissue, not a stroke.
    const ringJit = 0.58 + 0.42 * Math.sin(th * 7.0 + 1.1) + 0.24 * Math.sin(th * 17.0 - 2.3)
      + 0.14 * Math.sin(th * 31.0 + 0.7);
    const ringW = 0.80 + 0.60 * Math.sin(th * 5.0 + 0.4) + 0.26 * Math.sin(th * 13.0 - 1.2);
    const ring = Math.exp(-sq((r - R_PUPIL - 0.9) / ringW)) * 2.35 * clamp(ringJit, 0.06, 1.45);
    // asymmetric inner bloom: generous on the lit upper-left, tight elsewhere
    const bloomDir = 0.35 + 0.85 * smoothstep(0.6, -1.4, th + 1.9);
    const bloom = Math.exp(-Math.max(0, r - R_PUPIL - 1.6) / 17.0) * 0.26 * bloomDir *
      smoothstep(R_PUPIL, R_PUPIL + 9, r);
    ir_r *= 1 + ring + bloom * 1.04;
    ir_g *= 1 + ring * 0.80 + bloom * 0.60;
    ir_b *= 1 + ring * 0.56 + bloom * 0.32;
    // the iris rolls over the margin: a real shadow just outside the ring
    const roll = smoothstep(R_PUPIL + 16.0, R_PUPIL + 1.0, r) * 0.55;
    ir_r *= 1 - roll * 0.72; ir_g *= 1 - roll * 0.74; ir_b *= 1 - roll * 0.70;

    const irMask = smoothstep(R_IRIS + 2.5, R_IRIS - 3.0, r) * smoothstep(R_PUPIL - 2.0, R_PUPIL + 1.5, r);
    if (irMask > 0) {
      sc_r = mix(sc_r, ir_r, irMask);
      sc_g = mix(sc_g, ir_g, irMask);
      sc_b = mix(sc_b, ir_b, irMask);
    }
  }

  // ============ pupil: a true black circle with a hard pigment margin
  let pupilMask = 0;
  if (r < R_PUPIL + 7.0) {
    // hard edge: only ~1.5 px of AA, so the margin reads crisp at any zoom
    const pm = smoothstep(R_PUPIL + 0.5, R_PUPIL - 1.2, r);
    const ruff = Math.exp(-sq((r - R_PUPIL - 0.2) / 3.2)) * 0.92;
    const core = 0.00030 + 0.0016 * smoothstep(0.0, R_PUPIL, r) * n2 * n2;
    sc_r = mix(sc_r, core, pm) * (1 - ruff * pm);
    sc_g = mix(sc_g, core * 1.08, pm) * (1 - ruff * pm);
    sc_b = mix(sc_b, core * 1.62, pm) * (1 - ruff * pm);
    pupilMask = pm;
  }

  // ============ lid rim light on the inner edge; upper strong, lower thin
  const rimU = Math.exp(-sq(du / 2.0));
  const rimL = Math.exp(-sq(dl / 1.25));
  const sideL = 0.20 + 1.05 * smoothstep(0.95, -1.0, tRaw);
  const rimUW = rimU * sideL * 1.80;
  const rimLW = rimL * (0.30 + 0.52 * smoothstep(-1.0, 0.9, tRaw)) * 0.62;
  sc_r += rimUW * 2.75 + rimLW * 0.92;
  sc_g += rimUW * 1.72 + rimLW * 0.48;
  sc_b += rimUW * 0.84 + rimLW * 0.29;

  // ============ cornea
  // Speculars live on the iris/cornea surface, so they are clipped hard at the
  // pupil margin: the hole must stay black, and the edge must stay crisp.
  const specGate = 1 - pupilMask;
  // main specular, upper left of the iris: crisp, elongated, committed edge
  {
    const hx = ex + 128.0, hy = ey + 168.0;
    const ca = Math.cos(-0.55), sa = Math.sin(-0.55);
    const u = (ca * hx - sa * hy) / 30.0;
    const v = (sa * hx + ca * hy) / 13.0;
    const r4 = Math.pow(Math.abs(u), 3.4) + Math.pow(Math.abs(v), 3.4);
    const body = Math.exp(-r4 * 1.55);
    const halo = Math.exp(-(u * u + v * v) * 1.15) * 0.16;
    sc_r += (body * 14.0 + halo * 0.75) * specGate;
    sc_g += (body * 13.1 + halo * 0.73) * specGate;
    sc_b += (body * 12.0 + halo * 0.70) * specGate;
  }
  // small lower-right specular, out on the iris
  {
    const gx1 = ex - 158.0, gy1 = ey - 140.0;
    const ca2 = Math.cos(0.72), sa2 = Math.sin(0.72);
    const u = (ca2 * gx1 - sa2 * gy1) / 14.0;
    const v = (sa2 * gx1 + ca2 * gy1) / 6.5;
    const body = Math.exp(-(Math.pow(Math.abs(u), 3.2) + Math.pow(Math.abs(v), 3.2)) * 1.7);
    sc_r += body * 7.0 * specGate; sc_g += body * 6.5 * specGate; sc_b += body * 6.0 * specGate;
  }
  // wet sheen across the whole eye (kept off the pupil, or the hole goes milky)
  const sheen = 0.055 * (0.42 + 0.80 * smoothstep(1.05, -1.25, (ex + ey * 1.0) / A)) * (1 - 0.94 * pupilMask);
  sc_r += sheen * 1.0; sc_g += sheen * 1.05; sc_b += sheen * 1.22;

  // corner flare: light squeezed out where the lids just parted
  {
    // squeezed to a slit exactly at the canthal apex, not a blob 25 px inside it
    const cxx = Math.abs(ex) - A * 0.985;
    const wide = Math.exp(-sq(cxx / 26.0)) * Math.exp(-sq(ey / 9.0));
    const sharp = Math.exp(-sq(cxx / 9.0)) * Math.exp(-sq(ey / 2.6));
    sc_r += wide * 0.16 + sharp * 1.10;
    sc_g += wide * 0.09 + sharp * 0.62;
    sc_b += wide * 0.05 + sharp * 0.32;
  }
  // canthal crease: a dark pinch right at each corner, so the almond closes to a
  // point instead of dissolving into the flare above
  {
    const cxx = Math.abs(ex) - A * 0.999;
    const pinch = Math.exp(-sq(cxx / 22.0)) * smoothstep(0.55, 1.0, Math.abs(tRaw));
    sc_r *= 1 - 0.88 * pinch; sc_g *= 1 - 0.88 * pinch; sc_b *= 1 - 0.82 * pinch;
  }
  // tiny stray sparks
  for (let s = 0; s < 5; s++) {
    const side = s < 2 ? -1 : s < 4 ? 1 : 1;
    const sx2 = side * A * (0.72 + h2i(s, 1, 5) * 0.22);
    const sy2 = (h2i(s, 2, 6) - 0.5) * 0.22 * B;
    const dd = Math.hypot(ex - sx2, ey - sy2);
    if (dd < 18) {
      const qv = Math.exp(-sq(dd / 1.7)) * (0.5 + h2i(s, 3, 7) * 1.4);
      sc_r += qv * 1.6; sc_g += qv * 1.3; sc_b += qv * 0.95;
    }
  }

  // ============ lashes, applied last: they sit in front of the cornea, so
  // nothing — specular, bloom, rim light — may paint over them.
  const lsh = bilArr(LSH, SH, x * (SH / W), y * (SH / W));
  const lcore = bilArr(LX, W, x, y);
  if (lsh > 0.002) {
    const m = 1 - 0.86 * clamp(lsh * 1.9, 0, 1);
    sc_r *= m; sc_g *= m; sc_b *= m;
  }
  if (lcore > 0.004) {
    const m = 1 - 0.985 * clamp(lcore, 0, 1);
    sc_r *= m; sc_g *= m; sc_b *= m;
  }

  out[0] = mix(out[0], sc_r, inside);
  out[1] = mix(out[1], sc_g, inside);
  out[2] = mix(out[2], sc_b, inside);

  const gk = 1 + (gr - 0.5) * 0.05 + (n2 - 0.5) * 0.055;
  out[0] *= gk; out[1] *= gk; out[2] *= gk;
  lastPupil = pupilMask * inside;
  return out;
}

// ---------------------------------------------------------------- filmic
function tonemap(x: number): number {
  const v = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
  return v < 0 ? 0 : v;
}
function toSRGB(v: number): number {
  return Math.pow(clamp(v, 0, 1), 1 / 2.2);
}

// ---------------------------------------------------------------- render
function render(): Uint8Array {
  const buf = new Uint8Array(W * H * 3);
  const EXP = 1.22;
  const hdr = new Float32Array(W * H * 3);
  const pmask = new Float32Array(W * H);
  for (let py = 0; py < H; py++) {
    for (let px = 0; px < W; px++) {
      let R = 0, G = 0, Bl = 0, pmAcc = 0;
      for (let s = 0; s < 4; s++) {
        const jx = h2i(px, py, 17 + s), jy = h2i(px, py, 33 + s);
        const x = px + (s & 1) * 0.5 + 0.25 + (jx - 0.5) * 0.5;
        const y = py + (s >> 1) * 0.5 + 0.25 + (jy - 0.5) * 0.5;
        samplePixel(x, y, col, h2i(px, py, 91));
        R += col[0]; G += col[1]; Bl += col[2];
        pmAcc += lastPupil;
      }
      const o = (py * W + px) * 3;
      pmask[py * W + px] = pmAcc * 0.25;
      hdr[o] = R * 0.25 * EXP;
      hdr[o + 1] = G * 0.25 * EXP;
      hdr[o + 2] = Bl * 0.25 * EXP;
    }
  }

  // ---- bloom: bright pass, downsample, blur, add back (highlights bleed, never clip flat)
  const BD = 8; // downsample factor
  const BW2 = W / BD, BH2 = H / BD;
  const bright = new Float32Array(BW2 * BH2 * 3);
  for (let by = 0; by < BH2; by++) {
    for (let bx = 0; bx < BW2; bx++) {
      let r = 0, g = 0, b = 0;
      for (let j = 0; j < BD; j++) {
        const row = (by * BD + j) * W;
        for (let i = 0; i < BD; i++) {
          const o = (row + bx * BD + i) * 3;
          r += hdr[o]; g += hdr[o + 1]; b += hdr[o + 2];
        }
      }
      const inv = 1 / (BD * BD);
      r *= inv; g *= inv; b *= inv;
      const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      const k = l > 0.60 ? Math.min(1, (l - 0.60) / 1.4) : 0;
      const o = (by * BW2 + bx) * 3;
      bright[o] = r * k; bright[o + 1] = g * k; bright[o + 2] = b * k;
    }
  }
  // separable box blur, three passes at increasing radius => smooth glow
  const blurR = (src: Float32Array, w: number, h: number, r: number) => {
    const t = new Float32Array(src.length);
    const out2 = new Float32Array(src.length);
    const rad = r | 0;
    for (let y = 0; y < h; y++) {
      for (let ch = 0; ch < 3; ch++) {
        let sum = 0;
        for (let x = -rad; x <= rad; x++) {
          const xx = x < 0 ? 0 : x >= w ? w - 1 : x;
          sum += src[(y * w + xx) * 3 + ch];
        }
        const inv = 1 / (rad * 2 + 1);
        for (let x = 0; x < w; x++) {
          t[(y * w + x) * 3 + ch] = sum * inv;
          const xa = x - rad < 0 ? 0 : x - rad;
          const xr = x + rad + 1 >= w ? w - 1 : x + rad + 1;
          sum += src[(y * w + xr) * 3 + ch] - src[(y * w + xa) * 3 + ch];
        }
      }
    }
    for (let x = 0; x < w; x++) {
      for (let ch = 0; ch < 3; ch++) {
        let sum = 0;
        for (let y = -rad; y <= rad; y++) {
          const yy = y < 0 ? 0 : y >= h ? h - 1 : y;
          sum += t[(yy * w + x) * 3 + ch];
        }
        const inv = 1 / (rad * 2 + 1);
        for (let y = 0; y < h; y++) {
          out2[(y * w + x) * 3 + ch] = sum * inv;
          const ya = y - rad < 0 ? 0 : y - rad;
          const yr = y + rad + 1 >= h ? h - 1 : y + rad + 1;
          sum += t[(yr * w + x) * 3 + ch] - t[(ya * w + x) * 3 + ch];
        }
      }
    }
    return out2;
  };
  const bl1 = blurR(bright, BW2, BH2, 4);
  const bl2 = blurR(bl1, BW2, BH2, 9);

  for (let py = 0; py < H; py++) {
    for (let px = 0; px < W; px++) {
      const o = (py * W + px) * 3;
      // bilinear upsample of the two blur levels
      const fx = px / BD - 0.5, fy = py / BD - 0.5;
      const ix = Math.min(BW2 - 2, Math.max(0, Math.floor(fx)));
      const iy = Math.min(BH2 - 2, Math.max(0, Math.floor(fy)));
      const tx = Math.min(1, Math.max(0, fx - ix)), ty = Math.min(1, Math.max(0, fy - iy));
      const p00 = (iy * BW2 + ix) * 3, p10 = p00 + 3, p01 = p00 + BW2 * 3, p11 = p01 + 3;
      const bl = new Float32Array(3);
      for (let c = 0; c < 3; c++) {
        const a1 = mix(bl1[p00 + c], bl1[p10 + c], tx);
        const b1 = mix(bl1[p01 + c], bl1[p11 + c], tx);
        const a2 = mix(bl2[p00 + c], bl2[p10 + c], tx);
        const b2 = mix(bl2[p01 + c], bl2[p11 + c], tx);
        bl[c] = mix(a1, b1, ty) * 0.42 + mix(a2, b2, ty) * 0.62;
      }

      let R = hdr[o] + bl[0] * (1 - 0.90 * pmask[py * W + px]);
      let G = hdr[o + 1] + bl[1] * (1 - 0.90 * pmask[py * W + px]);
      let Bl = hdr[o + 2] + bl[2] * (1 - 0.90 * pmask[py * W + px]);

      // colour separation: push saturation about the luminance axis
      const lum = 0.2126 * R + 0.7152 * G + 0.0722 * Bl;
      const sat = 1.30;
      R = lum + (R - lum) * sat;
      G = lum + (G - lum) * sat;
      Bl = lum + (Bl - lum) * sat;

      let r = toSRGB(tonemap(R));
      let g = toSRGB(tonemap(G));
      let b = toSRGB(tonemap(Bl));
      // deepen the blacks, hold the highlights
      r = (r - 0.020) / 0.980;
      g = (g - 0.020) / 0.980;
      b = (b - 0.022) / 0.978;
      const dth = (h2i(px, py, 3) - 0.5) * (1.2 / 255);
      buf[o] = clamp(Math.round((r + dth) * 255), 0, 255);
      buf[o + 1] = clamp(Math.round((g + dth) * 255), 0, 255);
      buf[o + 2] = clamp(Math.round((b + dth) * 255), 0, 255);
    }
  }
  return buf;
}

// ---------------------------------------------------------------- PNG writer
const CRCTAB = (() => {
  const t = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c;
  }
  return t;
})();

function crc32(buf: Uint8Array, start: number, len: number): number {
  let c = 0xffffffff;
  for (let i = start; i < start + len; i++) {
    c = CRCTAB[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const len = data.length;
  const out = new Uint8Array(12 + len);
  out[0] = (len >>> 24) & 0xff; out[1] = (len >>> 16) & 0xff;
  out[2] = (len >>> 8) & 0xff; out[3] = len & 0xff;
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  const c = crc32(out, 4, 4 + len); // type + data, NOT the length field
  out[8 + len] = (c >>> 24) & 0xff;
  out[9 + len] = (c >>> 16) & 0xff;
  out[10 + len] = (c >>> 8) & 0xff;
  out[11 + len] = c & 0xff;
  return out;
}

function paeth(a: number, b: number, c: number): number {
  const p = a + b - c;
  const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

function encodePNG(rgb: Uint8Array): Uint8Array {
  const bpp = 3;
  const stride = W * bpp;
  const raw = new Uint8Array((stride + 1) * H);
  const cand = new Uint8Array(stride);
  const best = new Uint8Array(stride);
  for (let y = 0; y < H; y++) {
    const rowIn = y * stride;
    let bestScore = Infinity, bestF = 0;
    for (let f = 0; f < 5; f++) {
      let score = 0;
      for (let i = 0; i < stride; i++) {
        const x = rgb[rowIn + i];
        const a = i >= bpp ? rgb[rowIn + i - bpp] : 0;
        const b = y > 0 ? rgb[rowIn - stride + i] : 0;
        const c = i >= bpp && y > 0 ? rgb[rowIn - stride + i - bpp] : 0;
        let v: number;
        switch (f) {
          case 0: v = x; break;
          case 1: v = (x - a) & 0xff; break;
          case 2: v = (x - b) & 0xff; break;
          case 3: v = (x - ((a + b) >> 1)) & 0xff; break;
          default: v = (x - paeth(a, b, c)) & 0xff; break;
        }
        cand[i] = v;
        const sv = v < 128 ? v : 256 - v;
        score += sv;
      }
      if (score < bestScore) {
        bestScore = score;
        bestF = f;
        best.set(cand);
      }
    }
    const ro = y * (stride + 1);
    raw[ro] = bestF;
    raw.set(best.subarray(0, stride), ro + 1);
  }

  // Pick the smallest of a few deflate strategies, so we keep real headroom
  // under the 1 MiB cap instead of shipping a file that only just fits.
  let z = zlib.deflateSync(raw, { level: 9, memLevel: 9, strategy: zlib.constants.Z_DEFAULT_STRATEGY });
  for (const strat of [zlib.constants.Z_FILTERED, zlib.constants.Z_RLE]) {
    const cand = zlib.deflateSync(raw, { level: 9, memLevel: 9, strategy: strat });
    if (cand.length < z.length) z = cand;
  }
  const ihdr = new Uint8Array(13);
  ihdr[0] = (W >>> 24) & 0xff; ihdr[1] = (W >>> 16) & 0xff; ihdr[2] = (W >>> 8) & 0xff; ihdr[3] = W & 0xff;
  ihdr[4] = (H >>> 24) & 0xff; ihdr[5] = (H >>> 16) & 0xff; ihdr[6] = (H >>> 8) & 0xff; ihdr[7] = H & 0xff;
  ihdr[8] = 8;
  ihdr[9] = 2;
  ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;

  const sig = Uint8Array.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const parts = [
    sig,
    chunk("IHDR", ihdr),
    chunk("IDAT", new Uint8Array(z)),
    chunk("IEND", new Uint8Array(0)),
  ];
  let total = 0;
  for (const p of parts) total += p.length;
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}

// ---------------------------------------------------------------- main
const outDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
const outPath = path.join(outDir, "entry.png");
fs.mkdirSync(outDir, { recursive: true });

const tStart = Date.now();
const rgb = render();
const tRender = Date.now() - tStart;
const tEnc0 = Date.now();
const png = encodePNG(rgb);
const tEnc = Date.now() - tEnc0;

let prevHash = "";
try {
  const prev = fs.readFileSync(outPath);
  prevHash = crypto.createHash("sha256").update(prev).digest("hex");
} catch { /* first run */ }

fs.writeFileSync(outPath, png);
const newHash = crypto.createHash("sha256").update(png).digest("hex");

let sumL = 0, n09 = 0, n05 = 0, sumS = 0;
for (let i = 0; i < W * H; i++) {
  const r = rgb[i * 3] / 255, g = rgb[i * 3 + 1] / 255, b = rgb[i * 3 + 2] / 255;
  const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  sumL += l;
  if (l > 0.9) n09++;
  if (l > 0.5) n05++;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
  sumS += mx > 0.002 ? (mx - mn) / mx : 0;
}
const N = W * H;
console.log(JSON.stringify({
  file: outPath,
  bytes: png.length,
  limit: 1048576,
  renderMs: tRender,
  encodeMs: tEnc,
  totalMs: Date.now() - tStart,
  meanLum: +(sumL / N).toFixed(4),
  fracAbove09: +(n09 / N).toFixed(4),
  fracAbove05: +(n05 / N).toFixed(4),
  meanSat: +(sumS / N).toFixed(4),
  deterministic: prevHash ? newHash === prevHash : "first run",
  sha256: newHash.slice(0, 16),
}, null, 2));
