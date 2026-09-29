import { base64ToF32 } from "../base64";
import { softmax } from "../math";

export type CnnOp =
  | { op: "conv"; in_c: number; out_c: number; k: number; pad: number; w: string; b: string }
  | { op: "relu" }
  | { op: "maxpool2" }
  | { op: "flatten" }
  | { op: "linear"; in: number; out: number; w: string; b: string }
  | { op: "softmax" };

export interface CnnBlob {
  ops: CnnOp[];
}

/**
 * Ops-list interpreter for the exported CNN (conv 3×3/stride1/pad1, ReLU,
 * 2×2 maxpool, flatten, linear, softmax). BatchNorm was folded into conv/linear
 * weights at export time. Input: 784 floats ([0,1], CHW 1×28×28).
 */
export function createCnnEngine(blob: CnnBlob): (x: Float32Array) => number[] {
  const ops = blob.ops.map((o) => {
    if (o.op === "conv" || o.op === "linear") {
      const w = base64ToF32(o.w);
      const b = base64ToF32(o.b);
      if (o.op === "conv" && w.length !== o.out_c * o.in_c * o.k * o.k)
        throw new Error("conv w size mismatch");
      if (o.op === "linear" && w.length !== o.out * o.in)
        throw new Error("linear w size mismatch");
      return { ...o, w, b } as CnnOp & { w: Float32Array; b: Float32Array };
    }
    return o;
  });

  return function predict(input: Float32Array): number[] {
    let data = input;          // current activation (CHW or flat after flatten)
    let C = 1, H = 28, W = 28; // current spatial dims
    for (const o of ops as any[]) {
      switch (o.op) {
        case "conv": {
          const { w, b, in_c, out_c, k, pad } = o;
          const OH = H, OW = W; // stride 1, pad 1, k 3 keep size
          const out = new Float32Array(out_c * OH * OW);
          for (let oc = 0; oc < out_c; oc++) {
            const wo = oc * in_c * k * k;
            const yo = oc * OH * OW;
            const bias = b[oc];
            for (let oy = 0; oy < OH; oy++) {
              for (let ox = 0; ox < OW; ox++) {
                let acc = bias;
                for (let ic = 0; ic < in_c; ic++) {
                  const xo = ic * H * W;
                  const wi = wo + ic * k * k;
                  for (let ky = 0; ky < k; ky++) {
                    const iy = oy + ky - pad;
                    if (iy < 0 || iy >= H) continue;
                    const rowI = xo + iy * W;
                    const rowW = wi + ky * k;
                    for (let kx = 0; kx < k; kx++) {
                      const ix = ox + kx - pad;
                      if (ix < 0 || ix >= W) continue;
                      acc += data[rowI + ix] * w[rowW + kx];
                    }
                  }
                }
                out[yo + oy * OW + ox] = acc;
              }
            }
          }
          data = out;
          C = out_c;
          break;
        }
        case "relu": {
          for (let i = 0; i < data.length; i++) if (data[i] < 0) data[i] = 0;
          break;
        }
        case "maxpool2": {
          const OH = H >> 1, OW = W >> 1;
          const out = new Float32Array(C * OH * OW);
          for (let c = 0; c < C; c++) {
            const xo = c * H * W;
            const yo = c * OH * OW;
            for (let oy = 0; oy < OH; oy++) {
              for (let ox = 0; ox < OW; ox++) {
                let m = -Infinity;
                const x0 = xo + (oy << 1) * W + (ox << 1);
                if (data[x0] > m) m = data[x0];
                if (data[x0 + 1] > m) m = data[x0 + 1];
                if (data[x0 + W] > m) m = data[x0 + W];
                if (data[x0 + W + 1] > m) m = data[x0 + W + 1];
                out[yo + oy * OW + ox] = m;
              }
            }
          }
          data = out;
          H = OH;
          W = OW;
          break;
        }
        case "flatten":
          // CHW memory layout is already the flat order PyTorch's flatten uses
          break;
        case "linear": {
          const { w, b, in: inDim, out: outDim } = o;
          const outArr = new Float32Array(outDim);
          for (let oI = 0; oI < outDim; oI++) {
            let acc = b[oI];
            const wo = oI * inDim;
            for (let i = 0; i < inDim; i++) acc += w[wo + i] * data[i];
            outArr[oI] = acc;
          }
          data = outArr;
          C = outDim;
          H = 1;
          W = 1;
          break;
        }
        case "softmax":
          return softmax(data);
        default:
          throw new Error(`unknown op ${o.op}`);
      }
    }
    return softmax(data);
  };
}
