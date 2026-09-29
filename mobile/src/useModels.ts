import { useCallback, useRef, useState } from "react";
import { Asset } from "expo-asset";
import { loadModel } from "../../core/src/models";
import type { DigitModel } from "../../core/src/models";

export type ModelKind = "svc" | "mlp" | "cnn";
export type LoadState = "idle" | "loading" | "ready" | "error";

export interface ModelDef {
  kind: ModelKind;
  name: string;
  tag: string;
  desc: string;
  recommended?: boolean;
}

export const MODEL_DEFS: ModelDef[] = [
  { kind: "svc", name: "SVM", tag: "Classical ML", desc: "RBF-kernel SVM" },
  { kind: "mlp", name: "MLP", tag: "Neural Net", desc: "784→512→256→10" },
  { kind: "cnn", name: "CNN", tag: "Deep Learning", desc: "2 conv blocks", recommended: true },
];

const MODULES: Record<ModelKind, number> = {
  svc: require("../assets/models/svc_mnist.dlmodel"),
  mlp: require("../assets/models/mlp_mnist.dlmodel"),
  cnn: require("../assets/models/cnn_mnist.dlmodel"),
};

async function fetchModelJson(kind: ModelKind): Promise<any> {
  const asset = Asset.fromModule(MODULES[kind]);
  if (!asset.downloaded) await asset.downloadAsync();
  const res = await fetch(asset.localUri || asset.uri);
  return res.json();
}

/** Lazy, cached model registry backed by bundled assets. */
export function useModels() {
  const [states, setStates] = useState<Record<ModelKind, LoadState>>({
    svc: "idle",
    mlp: "idle",
    cnn: "idle",
  });
  const [models, setModels] = useState<Partial<Record<ModelKind, DigitModel>>>({});
  const promises = useRef<Partial<Record<ModelKind, Promise<DigitModel>>>>({});

  const load = useCallback((kind: ModelKind): Promise<DigitModel> => {
    const cached = promises.current[kind];
    if (cached) return cached;
    const p = fetchModelJson(kind)
      .then((json) => {
        const m = loadModel(json);
        setModels((prev) => ({ ...prev, [kind]: m }));
        setStates((prev) => ({ ...prev, [kind]: "ready" }));
        return m;
      })
      .catch((err) => {
        setStates((prev) => ({ ...prev, [kind]: "error" }));
        promises.current[kind] = undefined;
        throw err;
      });
    promises.current[kind] = p;
    setStates((prev) => ({ ...prev, [kind]: "loading" }));
    return p;
  }, []);

  const accuracy = useCallback((kind: ModelKind): number | null => {
    const m = models[kind];
    const acc = m?.metrics?.test_accuracy ?? m?.metrics?.accuracy;
    return typeof acc === "number" ? acc : null;
  }, [models]);

  return { states, models, load, accuracy };
}
