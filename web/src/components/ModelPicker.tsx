import type { LoadState, ModelDef, ModelKind } from "../hooks/useModels";

interface Props {
  defs: ModelDef[];
  selected: ModelKind;
  states: Record<ModelKind, LoadState>;
  accuracy: (k: ModelKind) => number | null;
  onSelect: (k: ModelKind) => void;
}

const ICONS: Record<ModelKind, JSX.Element> = {
  svc: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
      <circle cx="8" cy="16" r="5" />
      <circle cx="16" cy="8" r="5" />
      <path d="M12 12l3-3" strokeDasharray="1.5 2" />
    </svg>
  ),
  mlp: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
      <circle cx="4" cy="12" r="2" />
      <circle cx="12" cy="5" r="2" />
      <circle cx="12" cy="12" r="2" />
      <circle cx="12" cy="19" r="2" />
      <circle cx="20" cy="12" r="2" />
      <path d="M6 12l4-6M6 12l4 0M6 12l4 6M14 5l4 6M14 19l4-6M14 12l4 0" />
    </svg>
  ),
  cnn: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
      <rect x="3" y="3" width="7" height="7" rx="1.5" />
      <rect x="14" y="3" width="7" height="7" rx="1.5" />
      <rect x="8.5" y="14" width="7" height="7" rx="1.5" />
      <path d="M6.5 10v2h11v-2M12 14v-2" strokeDasharray="1.5 2" />
    </svg>
  ),
};

export default function ModelPicker({ defs, selected, states, accuracy, onSelect }: Props) {
  return (
    <div className="model-picker" role="radiogroup" aria-label="Model selection">
      {defs.map((d) => {
        const state = states[d.kind];
        const acc = accuracy(d.kind);
        const isSel = selected === d.kind;
        return (
          <button
            key={d.kind}
            role="radio"
            aria-checked={isSel}
            className={`model-card${isSel ? " selected" : ""}${state === "error" ? " error" : ""}`}
            onClick={() => onSelect(d.kind)}
            type="button"
          >
            <div className="model-card-top">
              <span className="model-icon">{ICONS[d.kind]}</span>
              <span className={`model-status ${state}`}>
                {state === "ready" ? "●" : state === "loading" ? "◌" : state === "error" ? "✕" : "○"}
              </span>
            </div>
            <div className="model-name">
              {d.name}
              {d.recommended && <span className="model-badge">best</span>}
            </div>
            <div className="model-tag">{d.tag}</div>
            <div className="model-desc">{d.desc}</div>
            <div className="model-acc">
              {acc !== null ? `${(acc * 100).toFixed(2)}% test acc` : `${d.family}`}
            </div>
          </button>
        );
      })}
    </div>
  );
}
