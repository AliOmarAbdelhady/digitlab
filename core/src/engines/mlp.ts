import { base64ToF32 } from "../base64";
import { softmax } from "../math";

export interface MlpLayerBlob {
  in: number;
  out: number;
  w: string; // float32, [out, in] row-major
  b: string; // float32, [out]
}

export interface MlpBlob {
  activation: "relu";
  layers: MlpLayerBlob[];
}

/** Dense-only MLP: ReLU between layers, softmax on logits. BatchNorm was folded
 *  into the linear weights at export time. */
export function createMlpEngine(blob: MlpBlob): (x: Float32Array) => number[] {
  const layers = blob.layers.map((L) => ({
    w: base64ToF32(L.w),
    b: base64ToF32(L.b),
    in: L.in,
    out: L.out,
  }));
  for (const L of layers) {
    if (L.w.length !== L.in * L.out || L.b.length !== L.out)
      throw new Error("mlp layer size mismatch");
  }
  // one scratch buffer per layer output
  const bufs = layers.map((L) => new Float32Array(L.out));
  const inputBuf = new Float32Array(layers[0].in);

  return function predict(input: Float32Array): number[] {
    inputBuf.set(input);
    for (let li = 0; li < layers.length; li++) {
      const L = layers[li];
      const src = li === 0 ? inputBuf : bufs[li - 1];
      const dst = bufs[li];
      const last = li === layers.length - 1;
      for (let o = 0; o < L.out; o++) {
        let acc = L.b[o];
        const wo = o * L.in;
        for (let i = 0; i < L.in; i++) acc += L.w[wo + i] * src[i];
        dst[o] = !last && acc < 0 ? 0 : acc; // ReLU between layers, raw logits at the end
      }
    }
    return softmax(bufs[bufs.length - 1]);
  };
}
