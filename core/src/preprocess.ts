/**
 * Drawing → MNIST-format 784-vector preprocessing.
 *
 * Reproduces the canonical MNIST preparation (LeCun et al.):
 *   1. rasterize the strokes (anti-aliased round brush) on a black field;
 *   2. crop to the ink bounding box;
 *   3. scale so the LONGER side becomes 20 px (aspect preserved, area-averaged);
 *   4. place into a 28×28 field so the ink's CENTER OF MASS sits at the field
 *      center (≈(14,14)) — exactly how MNIST digits are positioned;
 *   5. output floats in [0,1] (÷255), white digit on black.
 *
 * Extra robustness for finger/mouse input: stroke weight is normalized to
 * ~10.5% of the digit's longer side (the typical MNIST stroke ratio), so a large
 * thin drawing and a small thick drawing end up with comparable stroke width.
 *
 * Pure math on stroke polylines — no canvas/DOM — so the identical code runs in
 * browsers, Node and React Native.
 */

export interface Point {
  x: number;
  y: number;
}

export interface PreprocessResult {
  /** 784 floats in [0,1], CHW row-major 28×28 */
  vector: Float32Array;
  /** same as vector but bytes 0..255 (for previewing what the model sees) */
  image28: Uint8ClampedArray;
  /** ink bounding box in internal raster pixels */
  bbox: { x: number; y: number; w: number; h: number };
}

const R = 280; // internal raster resolution (10× the 28px field)
const OUT = 28;
const BOX = 20; // MNIST box size
const INK_THRESHOLD = 0.05;
const TARGET_STROKE_RATIO = 0.105; // stroke ÷ longer side, MNIST-typical
const MAX_STROKE_RATIO = 0.16;

function distToSegment(px: number, py: number, ax: number, ay: number,
                       bx: number, by: number): number {
  const dx = bx - ax;
  const dy = by - ay;
  const len2 = dx * dx + dy * dy;
  let t = len2 > 0 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const gx = ax + t * dx - px;
  const gy = ay + t * dy - py;
  return Math.sqrt(gx * gx + gy * gy);
}

/** Anti-aliased round-cap polyline rasterization, max-accumulated like ink. */
function rasterize(strokes: Point[][], scale: number, radius: number): Float32Array {
  const img = new Float32Array(R * R);
  for (const stroke of strokes) {
    for (let i = 0; i < stroke.length; i++) {
      const p = stroke[i];
      const q = i + 1 < stroke.length ? stroke[i + 1] : p;
      const ax = p.x * scale;
      const ay = p.y * scale;
      const bx = q.x * scale;
      const by = q.y * scale;
      const minX = Math.max(0, Math.floor(Math.min(ax, bx) - radius - 1));
      const maxX = Math.min(R - 1, Math.ceil(Math.max(ax, bx) + radius + 1));
      const minY = Math.max(0, Math.floor(Math.min(ay, by) - radius - 1));
      const maxY = Math.min(R - 1, Math.ceil(Math.max(ay, by) + radius + 1));
      for (let py = minY; py <= maxY; py++) {
        for (let px = minX; px <= maxX; px++) {
          const d = distToSegment(px + 0.5, py + 0.5, ax, ay, bx, by);
          const cov = radius + 0.5 - d; // 1px linear falloff
          if (cov > 0) {
            const v = cov > 1 ? 1 : cov;
            const idx = py * R + px;
            if (v > img[idx]) img[idx] = v;
          }
        }
      }
    }
  }
  return img;
}

function findBBox(img: Float32Array): { x: number; y: number; w: number; h: number } | null {
  let minX = R, minY = R, maxX = -1, maxY = -1;
  for (let y = 0; y < R; y++) {
    const row = y * R;
    for (let x = 0; x < R; x++) {
      if (img[row + x] > INK_THRESHOLD) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null;
  return { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };
}

function halveImg(src: Float32Array, w: number, h: number) {
  const nw = w >> 1;
  const nh = h >> 1;
  const out = new Float32Array(nw * nh);
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      const i0 = 2 * y * w + 2 * x;
      out[y * nw + x] = (src[i0] + src[i0 + 1] + src[i0 + w] + src[i0 + w + 1]) * 0.25;
    }
  }
  return { data: out, w: nw, h: nh };
}

/** Area-weighted box resample: each source pixel lands in exactly one target bin. */
function boxTo(src: Float32Array, w: number, h: number, tw: number, th: number): Float32Array {
  const sums = new Float64Array(tw * th);
  const cnts = new Float64Array(tw * th);
  for (let y = 0; y < h; y++) {
    const ty = Math.min(th - 1, Math.floor((y * th) / h));
    for (let x = 0; x < w; x++) {
      const tx = Math.min(tw - 1, Math.floor((x * tw) / w));
      const o = ty * tw + tx;
      sums[o] += src[y * w + x];
      cnts[o] += 1;
    }
  }
  const out = new Float32Array(tw * th);
  for (let i = 0; i < out.length; i++) out[i] = cnts[i] > 0 ? sums[i] / cnts[i] : 0;
  return out;
}

/** Crop → longer side 20px, progressive halving + final box resample. */
function resizeToBox(src: Float32Array, w: number, h: number) {
  const long = Math.max(w, h);
  const tw = Math.max(1, Math.round((w * BOX) / long));
  const th = Math.max(1, Math.round((h * BOX) / long));
  let cur = src;
  let cw = w;
  let ch = h;
  while ((cw >> 1) >= tw && (ch >> 1) >= th && cw > tw && ch > th) {
    const nx = halveImg(cur, cw, ch);
    cur = nx.data;
    cw = nx.w;
    ch = nx.h;
  }
  return { data: boxTo(cur, cw, ch, tw, th), w: tw, h: th };
}

function centerOfMass(data: Float32Array, w: number, h: number) {
  let s = 0;
  let sx = 0;
  let sy = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = data[y * w + x];
      if (v > 0) {
        s += v;
        sx += (x + 0.5) * v;
        sy += (y + 0.5) * v;
      }
    }
  }
  if (s <= 0) return { x: w / 2, y: h / 2 };
  return { x: sx / s, y: sy / s };
}

/**
 * @param strokes       polylines in canvas logical coordinates
 * @param canvasSize    logical canvas side length (square canvas assumed)
 * @param brushRadius   brush radius in canvas units (visual stroke width / 2)
 */
export function strokesToVector(
  strokes: Point[][],
  canvasSize: number,
  brushRadius: number,
): PreprocessResult | null {
  const clean = (strokes || []).filter((s) => s && s.length > 0);
  if (clean.length === 0) return null;

  const scale = R / canvasSize;
  let radius = Math.max(1.5, brushRadius * scale);
  let img = rasterize(clean, scale, radius);
  let bbox = findBBox(img);
  if (!bbox) return null;

  // Stroke-weight normalization: thicken if ink is thin RELATIVE to the digit
  // (e.g., a big digit drawn with a slim brush would otherwise become 1px in the
  // 20px box, much thinner than MNIST strokes).
  const longSide = Math.max(bbox.w, bbox.h);
  const target = longSide * TARGET_STROKE_RATIO;
  if (target > radius * 1.15) {
    radius = Math.min(target, longSide * MAX_STROKE_RATIO);
    img = rasterize(clean, scale, radius);
    bbox = findBBox(img)!;
  }

  // crop
  const crop = new Float32Array(bbox.w * bbox.h);
  for (let y = 0; y < bbox.h; y++) {
    const from = (bbox.y + y) * R + bbox.x;
    crop.set(img.subarray(from, from + bbox.w), y * bbox.w);
  }

  // resize to 20px box + center by center of mass in the 28×28 field
  const patch = resizeToBox(crop, bbox.w, bbox.h);
  const c = centerOfMass(patch.data, patch.w, patch.h);
  let ox = Math.round(OUT / 2 - c.x);
  let oy = Math.round(OUT / 2 - c.y);
  ox = Math.min(Math.max(ox, 0), OUT - patch.w);
  oy = Math.min(Math.max(oy, 0), OUT - patch.h);

  const vector = new Float32Array(OUT * OUT);
  for (let y = 0; y < patch.h; y++) {
    for (let x = 0; x < patch.w; x++) {
      const v = patch.data[y * patch.w + x];
      if (v > 0) vector[(oy + y) * OUT + (ox + x)] = v;
    }
  }
  const image28 = new Uint8ClampedArray(OUT * OUT);
  for (let i = 0; i < OUT * OUT; i++) image28[i] = Math.round(vector[i] * 255);
  return { vector, image28, bbox };
}

/** Identity helper for image inputs: 784 uint8 pixels → float vector [0,1]. */
export function image28ToVector(pixels: Uint8Array | Uint8ClampedArray): Float32Array {
  const out = new Float32Array(784);
  for (let i = 0; i < 784; i++) out[i] = pixels[i] / 255;
  return out;
}
