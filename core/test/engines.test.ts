import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { decodeBase64 } from "../src/base64";
import { strokesToVector, image28ToVector } from "../src/preprocess";
import { loadModel } from "../src/models";
import type { ModelJson } from "../src/models";

const ROOT = join(__dirname, "..", "..");
const fixtures = JSON.parse(
  readFileSync(join(ROOT, "core", "test", "fixtures", "fixtures.json"), "utf8"),
);

function loadJsonModel(kind: string) {
  const p = join(ROOT, "models", `${kind}_mnist.json`);
  return loadModel(JSON.parse(readFileSync(p, "utf8")) as ModelJson);
}

describe("base64 decoder", () => {
  it("round-trips binary blobs", () => {
    for (const [input, hex] of [
      ["", ""],
      ["AA==", "00"],
      ["/w==", "ff"],
      ["QUJD", "414243"],
      ["QUJDRA==", "41424344"],
    ] as const) {
      const out = decodeBase64(input);
      expect(Array.from(out).map((b) => b.toString(16).padStart(2, "0")).join("")).toBe(hex);
    }
  });
  it("handles whitespace and no padding", () => {
    expect(decodeBase64("QU JD\nRA").length).toBe(4);
    expect(Array.from(decodeBase64("QUJDRA")).length).toBe(4);
  });
});

describe("preprocess: strokes → 784", () => {
  const S = 300; // canvas size

  it("returns null for empty input", () => {
    expect(strokesToVector([], S, 12)).toBeNull();
    expect(strokesToVector([[]], S, 12)).toBeNull();
  });

  it("centers ink by center of mass regardless of where it was drawn", () => {
    const mkLine = (x: number): [{ x: number; y: number }[]] => [
      [
        { x, y: 40 },
        { x, y: S - 40 },
      ],
    ];
    for (const x of [60, 150, 240]) {
      const r = strokesToVector(mkLine(x), S, 10);
      expect(r).not.toBeNull();
      // center of mass must sit at the field center
      let sx = 0, sy = 0, s = 0;
      for (let i = 0; i < 784; i++) {
        const v = r!.vector[i];
        if (v > 0) {
          sx += ((i % 28) + 0.5) * v;
          sy += (Math.floor(i / 28) + 0.5) * v;
          s += v;
        }
      }
      expect(Math.abs(sx / s - 14)).toBeLessThanOrEqual(0.75);
      expect(Math.abs(sy / s - 14)).toBeLessThanOrEqual(0.75);
    }
  });

  it("scales the longer side into the 20px box and fills [0,1]", () => {
    // horizontal bar drawn wide
    const r = strokesToVector(
      [[{ x: 20, y: 150 }, { x: 280, y: 150 }]],
      S,
      12,
    )!;
    let minX = 28, maxX = -1, minY = 28, maxY = -1, maxV = 0;
    for (let i = 0; i < 784; i++) {
      const v = r.vector[i];
      if (v > 0.05) {
        const x = i % 28, y = Math.floor(i / 28);
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        if (v > maxV) maxV = v;
      }
    }
    expect(maxX - minX + 1).toBeGreaterThanOrEqual(18); // ≈20 box
    expect(maxX - minX + 1).toBeLessThanOrEqual(22);
    expect(maxY - minY + 1).toBeLessThanOrEqual(8); // thin stroke stays thin-ish
    expect(maxV).toBeGreaterThan(0.85);
  });

  it("image28ToVector divides by 255", () => {
    const v = image28ToVector(new Uint8Array(784).fill(255));
    expect(v[0]).toBeCloseTo(1.0);
    expect(v.length).toBe(784);
  });
});

for (const kind of ["svc", "mlp", "cnn"] as const) {
  describe(`${kind} engine parity vs python export`, () => {
    const model = loadJsonModel(kind);
    const images = decodeBase64(fixtures.images_u8);
    const n: number = fixtures.n;
    const exp = fixtures.models[kind];

    it("matches python predictions and probabilities", () => {
      let agree = 0;
      let maxDiff = 0;
      for (let i = 0; i < n; i++) {
        const x = image28ToVector(images.subarray(i * 784, (i + 1) * 784));
        const probs = model.predict(x);
        const am = probs.indexOf(Math.max(...probs));
        if (am === exp.preds[i]) agree++;
        for (let c = 0; c < 10; c++) {
          const d = Math.abs(probs[c] - exp.probs[i][c]);
          if (d > maxDiff) maxDiff = d;
        }
      }
      expect(agree / n).toBeGreaterThanOrEqual(0.996);
      expect(maxDiff).toBeLessThan(1e-3);
    }, 180_000);

    it("predictFull returns digit, probs and latency", () => {
      const x = image28ToVector(images.subarray(0, 784));
      const p = model.predictFull(x);
      expect(p.digit).toBeGreaterThanOrEqual(0);
      expect(p.digit).toBeLessThanOrEqual(9);
      expect(p.probs.length).toBe(10);
      expect(p.probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
      expect(p.latencyMs).toBeGreaterThanOrEqual(0);
    });
  });
}
