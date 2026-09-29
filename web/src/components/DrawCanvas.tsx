import { useEffect, useImperativeHandle, useRef, forwardRef } from "react";
import type { Point } from "@core/preprocess";

export type Stroke = Point[];

export interface CanvasApi {
  redraw: () => void;
  canvasSize: () => number;
}

interface Props {
  strokes: Stroke[];
  version: number;
  onStrokeCommit: (s: Stroke) => void;
  onInkChange: (hasInk: boolean) => void;
  apiRef?: React.MutableRefObject<CanvasApi | null>;
}

const CANVAS_CSS = 340; // logical square size (CSS px)
const BRUSH_RATIO = 0.045; // brush radius ÷ canvas size

/** Drawing surface — pointer/touch input, glowing ink, empty-state hint. */
const DrawCanvas = forwardRef<HTMLCanvasElement, Props>(function DrawCanvas(
  { strokes, version, onStrokeCommit, onInkChange, apiRef },
  ref,
) {
  const localRef = useRef<HTMLCanvasElement | null>(null);
  const activeStroke = useRef<Stroke | null>(null);
  const sizeRef = useRef(CANVAS_CSS);
  const commitRef = useRef(onStrokeCommit);
  commitRef.current = onStrokeCommit;

  const setCanvas = (el: HTMLCanvasElement | null) => {
    localRef.current = el;
    if (typeof ref === "function") ref(el);
    else if (ref) (ref as React.MutableRefObject<HTMLCanvasElement | null>).current = el;
  };

  const ctxOf = () => {
    const cv = localRef.current;
    if (!cv) return null;
    const dpr = window.devicePixelRatio || 1;
    const cssSize = cv.clientWidth || CANVAS_CSS;
    sizeRef.current = cssSize;
    const px = Math.round(cssSize * dpr);
    if (cv.width !== px) {
      cv.width = px;
      cv.height = px;
    }
    const ctx = cv.getContext("2d");
    if (!ctx) return null;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  };

  const brushRadius = () => sizeRef.current * BRUSH_RATIO;

  const paintStroke = (ctx: CanvasRenderingContext2D, stroke: Stroke, from = 0) => {
    if (stroke.length === 0) return;
    ctx.strokeStyle = "#eaf1ff";
    ctx.fillStyle = "#eaf1ff";
    ctx.lineWidth = brushRadius() * 2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.shadowColor = "rgba(96, 150, 255, 0.5)";
    ctx.shadowBlur = brushRadius() * 1.2;
    if (stroke.length === 1) {
      const p = stroke[0];
      ctx.beginPath();
      ctx.arc(p.x, p.y, brushRadius(), 0, Math.PI * 2);
      ctx.fill();
      return;
    }
    const start = Math.max(0, from - 1);
    ctx.beginPath();
    ctx.moveTo(stroke[start].x, stroke[start].y);
    for (let i = start + 1; i < stroke.length; i++) ctx.lineTo(stroke[i].x, stroke[i].y);
    ctx.stroke();
  };

  const redrawAll = () => {
    const ctx = ctxOf();
    if (!ctx) return;
    const size = sizeRef.current;
    ctx.shadowBlur = 0;
    ctx.clearRect(0, 0, size, size);
    for (const s of strokes) paintStroke(ctx, s);
    if (activeStroke.current && activeStroke.current.length > 0)
      paintStroke(ctx, activeStroke.current);
  };

  useEffect(() => {
    redrawAll();
  }, [version]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const onResize = () => redrawAll();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useImperativeHandle(apiRef, () => ({
    redraw: redrawAll,
    canvasSize: () => sizeRef.current,
  }));

  const getPos = (e: React.PointerEvent<HTMLCanvasElement>): Point => {
    const rect = e.currentTarget.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    activeStroke.current = [getPos(e)];
    const ctx = ctxOf();
    if (ctx) paintStroke(ctx, activeStroke.current);
    onInkChange(true);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const stroke = activeStroke.current;
    if (!stroke) return;
    const p = getPos(e);
    const last = stroke[stroke.length - 1];
    if (Math.hypot(p.x - last.x, p.y - last.y) < 1.5) return; // dedupe micro-moves
    const from = stroke.length;
    stroke.push(p);
    const ctx = ctxOf();
    if (ctx) paintStroke(ctx, stroke, from);
  };

  const finishStroke = () => {
    const s = activeStroke.current;
    activeStroke.current = null;
    if (s && s.length > 0) commitRef.current(s);
  };

  return (
    <div className="canvas-wrap">
      <canvas
        ref={setCanvas}
        className="draw-canvas"
        style={{ touchAction: "none" }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={finishStroke}
        onPointerCancel={finishStroke}
      />
      {strokes.length === 0 && (
        <div className="canvas-hint">
          <span className="canvas-hint-glyph">✎</span>
          <span>draw a digit here</span>
          <span className="canvas-hint-sub">0 – 9 · one or several strokes</span>
        </div>
      )}
    </div>
  );
});

DrawCanvas.displayName = "DrawCanvas";
export default DrawCanvas;
