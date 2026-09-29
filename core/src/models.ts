import { base64ToF32, base64ToU8 } from "./base64";
import { softmax, argmax } from "./math";

/** Portable model JSON produced by the Kaggle notebooks. */
export interface ModelJson {
  format: "digitlab-model";
  version: number;
  kind: "svc" | "mlp" | "cnn";
  created: string;
  framework?: string;
  classes: number[];
  metrics?: Record<string, any>;
  model: any; // kind-specific, see engines
}

export interface Prediction {
  digit: number;
  /** probabilities per class position (classes[i]); sums to 1 */
  probs: number[];
  latencyMs: number;
}

export interface DigitModel {
  kind: "svc" | "mlp" | "cnn";
  classes: number[];
  metrics?: Record<string, any>;
  /** x784: Float32Array of length 784, values in [0,1], white digit on black */
  predict(x784: Float32Array): number[];
  predictFull(x784: Float32Array): Prediction;
}

/** Wrap a raw predict into the Prediction interface (timing + argmax). */
function withMeta(kind: DigitModel["kind"], classes: number[],
                  predict: (x: Float32Array) => number[]): DigitModel {
  return {
    kind,
    classes,
    predict,
    predictFull(x784: Float32Array): Prediction {
      const t0 =
        typeof performance !== "undefined" ? performance.now() : Date.now();
      const probs = predict(x784);
      const t1 =
        typeof performance !== "undefined" ? performance.now() : Date.now();
      return { digit: classes[argmax(probs)], probs, latencyMs: t1 - t0 };
    },
  };
}

import { createSvcEngine } from "./engines/svc";
import { createMlpEngine } from "./engines/mlp";
import { createCnnEngine } from "./engines/cnn";

/** Parse a portable model JSON into a runnable engine. */
export function loadModel(json: ModelJson): DigitModel {
  if (json.format !== "digitlab-model") throw new Error("not a digitlab model");
  switch (json.kind) {
    case "svc":
      return withMeta("svc", json.classes, createSvcEngine(json.model));
    case "mlp":
      return withMeta("mlp", json.classes, createMlpEngine(json.model));
    case "cnn":
      return withMeta("cnn", json.classes, createCnnEngine(json.model));
    default:
      throw new Error(`unknown kind ${json.kind}`);
  }
}

/** Fetch + parse a model JSON (browser / Node 18+ fetch). */
export async function fetchModel(url: string): Promise<DigitModel> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to fetch ${url}: ${res.status}`);
  return loadModel(await res.json());
}

export { base64ToF32, base64ToU8 };
