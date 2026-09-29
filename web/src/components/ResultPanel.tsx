import { useEffect, useMemo, useState } from "react";
import type { Prediction } from "@core/models";

interface Props {
  result: Prediction | null;
  modelName: string;
  thinking: boolean;
}

function ConfidenceRing({ value }: { value: number }) {
  const R = 52;
  const C = 2 * Math.PI * R;
  const [anim, setAnim] = useState(0);
  useEffect(() => {
    const t = requestAnimationFrame(() => setAnim(value));
    return () => cancelAnimationFrame(t);
  }, [value]);
  return (
    <div className="ring-wrap">
      <svg viewBox="0 0 120 120" className="ring">
        <defs>
          <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#22d3ee" />
            <stop offset="100%" stopColor="#6366f1" />
          </linearGradient>
        </defs>
        <circle cx="60" cy="60" r={R} className="ring-track" />
        <circle
          cx="60"
          cy="60"
          r={R}
          className="ring-value"
          stroke="url(#ringGrad)"
          strokeDasharray={C}
          strokeDashoffset={C * (1 - anim)}
        />
      </svg>
      <div className="ring-label">
        <span className="ring-pct">{(value * 100).toFixed(1)}%</span>
        <span className="ring-sub">confidence</span>
      </div>
    </div>
  );
}

export default function ResultPanel({ result, modelName, thinking }: Props) {
  const top3 = useMemo(() => {
    if (!result) return [];
    return result.probs
      .map((p, i) => ({ digit: i, p }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 3);
  }, [result]);

  const [displayDigit, setDisplayDigit] = useState(0);
  useEffect(() => {
    if (result) setDisplayDigit(result.digit);
  }, [result]);

  if (thinking) {
    return (
      <div className="result-panel thinking" aria-live="polite">
        <div className="spinner" />
        <span>running {modelName} …</span>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="result-panel empty">
        <div className="result-ghost">?</div>
        <div className="result-empty-text">
          draw a digit, pick a model, press <b>Predict</b>
        </div>
      </div>
    );
  }

  return (
    <div className="result-panel" aria-live="polite">
      <div className="result-digit-block">
        <div className="result-digit" key={displayDigit}>
          {displayDigit}
        </div>
        <div className="result-model-chip">{modelName}</div>
      </div>
      <ConfidenceRing value={result.probs[result.digit] ?? 0} />
      <div className="result-bars">
        {top3.map((t, i) => (
          <div className="bar-row" key={t.digit}>
            <span className="bar-digit">{t.digit}</span>
            <div className="bar-track">
              <div
                className={`bar-fill${i === 0 ? " first" : ""}`}
                style={{ width: `${Math.max(2, t.p * 100)}%` }}
              />
            </div>
            <span className="bar-val">{(t.p * 100).toFixed(1)}%</span>
          </div>
        ))}
        <div className="latency-chip">⚡ {result.latencyMs.toFixed(1)} ms · on-device</div>
      </div>
    </div>
  );
}
