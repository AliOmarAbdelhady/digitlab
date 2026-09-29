import { base64ToF32, base64ToU8 } from "../base64";
import { softmax } from "../math";

export interface SvcBlob {
  gamma: number;
  C: number;
  n_sv: number;
  n_features: number; // 784
  class_offsets: number[]; // len 11
  sv_u8: string;    // n_sv × 784, pixel units 0..255
  dual_f32: string; // n_sv × 10
  intercept_f32: string; // 10
}

/**
 * RBF-SVM inference, mirroring the export in notebook 1:
 *   score_c(x) = Σ_i dual[i][c] · exp(−γ·‖x − sv_i/255‖²) + b_c
 * Support vectors stay uint8 (quantization error ≤ 1/510 per pixel — verified
 * in the notebook parity cell to keep 100% prediction agreement).
 */
export function createSvcEngine(blob: SvcBlob): (x: Float32Array) => number[] {
  const D = blob.n_features;
  const n = blob.n_sv;
  const sv = base64ToU8(blob.sv_u8);
  const dual = base64ToF32(blob.dual_f32);
  const b = base64ToF32(blob.intercept_f32);
  if (sv.length !== n * D) throw new Error("sv_u8 size mismatch");
  if (dual.length !== n * 10) throw new Error("dual_f32 size mismatch");
  const gamma = blob.gamma;
  const inv = 1 / 255;
  const scores = new Float64Array(10);

  return function predict(x: Float32Array): number[] {
    scores.fill(0);
    for (let i = 0; i < n; i++) {
      const so = i * D;
      const xo = i * 10;
      let d2 = 0;
      for (let j = 0; j < D; j++) {
        const s8 = sv[so + j];
        const xv = x[j];
        if (xv === 0 && s8 === 0) continue; // background pixel: contributes 0
        const d = xv - s8 * inv;
        d2 += d * d;
      }
      const k = Math.exp(-gamma * d2);
      for (let c = 0; c < 10; c++) scores[c] += dual[xo + c] * k;
    }
    for (let c = 0; c < 10; c++) scores[c] += b[c];
    return softmax(scores);
  };
}
