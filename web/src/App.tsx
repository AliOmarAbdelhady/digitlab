import { useCallback, useEffect, useRef, useState } from "react";
import DrawCanvas, { type CanvasApi, type Stroke } from "./components/DrawCanvas";
import ModelPicker from "./components/ModelPicker";
import Preview28 from "./components/Preview28";
import ResultPanel from "./components/ResultPanel";
import { MODEL_DEFS, useModels, type ModelKind } from "./hooks/useModels";
import { strokesToVector } from "@core/preprocess";
import type { Prediction } from "@core/models";

export default function App() {
  const { states, models, load, accuracy } = useModels();
  const [selected, setSelected] = useState<ModelKind>("cnn");
  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const [version, setVersion] = useState(0);
  const [result, setResult] = useState<Prediction | null>(null);
  const [preview, setPreview] = useState<Uint8ClampedArray | null>(null);
  const [thinking, setThinking] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const canvasApi = useRef<CanvasApi | null>(null);

  // eager-load the selected model (badge shows real accuracy once ready)
  useEffect(() => {
    load(selected).catch((e) => setToast(`failed to load model: ${e.message}`));
  }, [selected, load]);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 3500);
    return () => clearTimeout(t);
  }, [toast]);

  const hasInk = strokes.length > 0;

  const onStrokeCommit = useCallback((s: Stroke) => {
    setStrokes((prev) => [...prev, s]);
    setResult(null);
  }, []);

  const clearAll = useCallback(() => {
    setStrokes([]);
    setResult(null);
    setPreview(null);
    setVersion((v) => v + 1);
  }, []);

  const predictNow = useCallback(async (): Promise<Prediction | null> => {
    if (strokes.length === 0) {
      setToast("draw a digit first ✎");
      return null;
    }
    const canvasSize = canvasApi.current?.canvasSize() ?? 340;
    const pre = strokesToVector(strokes, canvasSize, canvasSize * 0.045);
    if (!pre) {
      setToast("ink not detected — draw a bigger digit");
      return null;
    }
    setPreview(pre.image28);
    setThinking(true);
    try {
      const model = await load(selected);
      const r = model.predictFull(pre.vector);
      setResult(r);
      return r;
    } catch (e: any) {
      setToast(`prediction failed: ${e?.message ?? e}`);
      return null;
    } finally {
      setThinking(false);
    }
  }, [strokes, selected, load]);

  // keyboard: Enter → predict, Escape → clear
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Enter") void predictNow();
      if (e.key === "Escape") clearAll();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [predictNow, clearAll]);

  // ---- automated test hook (used by browser e2e tests; harmless in normal use)
  // registered ONCE with a stable identity; per-render state flows through refs
  const predictRef = useRef(predictNow);
  predictRef.current = predictNow;
  const clearRef = useRef(clearAll);
  clearRef.current = clearAll;
  const stateRef = useRef<any>({});
  stateRef.current = {
    selected,
    strokes: strokes.length,
    hasInk: strokes.length > 0,
    result: result ? { digit: result.digit, latencyMs: result.latencyMs } : null,
    loaded: Object.fromEntries(MODEL_DEFS.map((d) => [d.kind, states[d.kind]])),
  };
  useEffect(() => {
    (window as any).__digitlab = {
      draw: async (s: Stroke[]) => {
        setStrokes(s);
        setVersion((v) => v + 1);
        setResult(null);
      },
      clear: () => clearRef.current(),
      predict: () => {
        // let React commit the pending draw() state before predicting
        return new Promise<any>((resolve) =>
          setTimeout(() => predictRef.current().then(resolve), 30),
        );
      },
      state: () => stateRef.current,
      select: (k: ModelKind) => setSelected(k),
    };
  }, []);

  const def = MODEL_DEFS.find((d) => d.kind === selected)!;

  return (
    <div className="app">
      <div className="bg-glow glow-a" />
      <div className="bg-glow glow-b" />

      <header className="header">
        <div className="logo">
          <span className="logo-mark">7</span>
          <div className="logo-text">
            <div className="logo-name">DigitLab</div>
            <div className="logo-sub">MNIST · three models · one canvas</div>
          </div>
        </div>
        <div className="header-note">all inference runs in your browser — no server</div>
      </header>

      <main className="main">
        <section className="panel models-panel" aria-label="Model selection">
          <div className="panel-title">
            <span className="panel-kicker">01</span> choose your model
          </div>
          <ModelPicker
            defs={MODEL_DEFS}
            selected={selected}
            states={states}
            accuracy={accuracy}
            onSelect={(k) => setSelected(k)}
          />
        </section>

        <div className="columns">
          <section className="panel canvas-panel" aria-label="Drawing canvas">
            <div className="panel-title">
              <span className="panel-kicker">02</span> draw a digit
            </div>
            <DrawCanvas
              strokes={strokes}
              version={version}
              onStrokeCommit={onStrokeCommit}
              onInkChange={() => setResult(null)}
              apiRef={canvasApi}
            />
            <div className="actions">
              <button className="btn ghost" onClick={clearAll} disabled={!hasInk} type="button">
                Clear
              </button>
              <button
                className="btn primary"
                onClick={() => void predictNow()}
                disabled={!hasInk || thinking}
                type="button"
              >
                {thinking ? "Predicting…" : `Predict with ${def.name}`}
              </button>
            </div>
          </section>

          <section className="panel result-side" aria-label="Prediction result">
            <div className="panel-title">
              <span className="panel-kicker">03</span> prediction
            </div>
            <ResultPanel
              result={result}
              modelName={def.name}
              thinking={thinking && !result}
            />
            <Preview28 image28={preview} />
          </section>
        </div>
      </main>

      <footer className="footer">
        trained on MNIST (60k) · Kaggle notebooks in the repo ·{" "}
        <span className="footer-dim">
          SVC / MLP / CNN exported to a portable JSON format, executed by a
          hand-written JS inference core
        </span>
      </footer>

      {toast && (
        <div className="toast" role="status">
          {toast}
        </div>
      )}
    </div>
  );
}
