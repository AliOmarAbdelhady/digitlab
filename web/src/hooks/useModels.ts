import { useCallback, useRef, useState } from "react";
import { fetchModel } from "@core/models";
import type { DigitModel } from "@core/models";

export type ModelKind = "svc" | "mlp" | "cnn";
export type LoadState = "idle" | "loading" | "ready" | "error";

export interface ModelDef {
  kind: ModelKind;
  name: string;
  tag: string;
  desc: string;
  family: string;
  recommended?: boolean;
}

export const MODEL_DEFS: ModelDef[] = [
  {
    kind: "svc",
    name: "SVM",
    tag: "Classical ML",
    desc: "RBF-kernel support vector machine",
    family: "scikit-learn SVC",
  },
  {
    kind: "mlp",
    name: "MLP",
    tag: "Neural Network",
    desc: "Fully-connected 784→512→256→10",
    family: "PyTorch MLP",
  },
  {
    kind: "cnn",
    name: "CNN",
    tag: "Deep Learning",
    desc: "2 conv blocks · 468k parameters",
    family: "PyTorch CNN",
    recommended: true,
  },
];

/** Lazy, cached model registry — models are fetched on first use. */
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
    const p = fetchModel(`models/${kind}_mnist.json`)
      .then((m) => {
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

  const accuracy = useCallback(
    (kind: ModelKind): number | null => {
      const m = models[kind];
      const acc = m?.metrics?.test_accuracy ?? m?.metrics?.accuracy;
      return typeof acc === "number" ? acc : null;
    },
    [models],
  );

  return { states, models, load, accuracy };
}
