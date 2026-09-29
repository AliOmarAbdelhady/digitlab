import React, { useMemo, useRef, useState } from "react";
import { PanResponder, StyleSheet, View } from "react-native";
import Svg, { Line, Polyline } from "react-native-svg";
import { theme } from "./theme";

export interface Pt {
  x: number;
  y: number;
}
export type Stroke = Pt[];

interface Props {
  size: number;
  strokes: Stroke[];
  onStrokeCommit: (s: Stroke) => void;
}

const BRUSH_RATIO = 0.045; // brush radius ÷ canvas size (matches web app)

/** Finger-drawing surface: PanResponder input + SVG ink rendering. */
export default function DrawCanvas({ size, strokes, onStrokeCommit }: Props) {
  const [live, setLive] = useState<Stroke | null>(null);
  const liveRef = useRef<Stroke | null>(null);
  const commitRef = useRef(onStrokeCommit);
  commitRef.current = onStrokeCommit;

  const pan = useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => true,
        onMoveShouldSetPanResponder: () => true,
        onPanResponderGrant: (e) => {
          const p = {
            x: e.nativeEvent.locationX,
            y: e.nativeEvent.locationY,
          };
          liveRef.current = [p];
          setLive([p]);
        },
        onPanResponderMove: (e) => {
          const s = liveRef.current;
          if (!s) return;
          const p = {
            x: Math.max(0, Math.min(size, e.nativeEvent.locationX)),
            y: Math.max(0, Math.min(size, e.nativeEvent.locationY)),
          };
          const last = s[s.length - 1];
          if (Math.hypot(p.x - last.x, p.y - last.y) < 2) return;
          s.push(p);
          setLive([...s]);
        },
        onPanResponderRelease: () => finish(),
        onPanResponderTerminate: () => finish(),
      }),
    [size],
  );

  const finish = () => {
    const s = liveRef.current;
    liveRef.current = null;
    setLive(null);
    if (s && s.length > 0) commitRef.current(s);
  };

  const brush = size * BRUSH_RATIO;
  const all = live ? [...strokes, live] : strokes;
  const toPts = (s: Stroke) => s.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");

  const gridLines = useMemo(() => {
    const els: React.ReactElement[] = [];
    const step = size / 10;
    for (let i = 1; i < 10; i++) {
      const pos = (step * i).toFixed(1);
      els.push(
        <Line key={`h${i}`} x1="0" y1={pos} x2={size} y2={pos} stroke="rgba(148,178,255,0.06)" strokeWidth="1" />,
        <Line key={`v${i}`} x1={pos} y1="0" x2={pos} y2={size} stroke="rgba(148,178,255,0.06)" strokeWidth="1" />,
      );
    }
    return els;
  }, [size]);

  return (
    <View style={[styles.wrap, { width: size, height: size, borderRadius: 18 }]}>
      <Svg width={size} height={size} style={styles.svg}>
        {gridLines}
        {all.map((s, i) => (
          <React.Fragment key={i}>
            <Polyline
              points={toPts(s)}
              stroke="rgba(96,150,255,0.28)"
              strokeWidth={brush * 3.2}
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
            <Polyline
              points={toPts(s)}
              stroke={theme.ink}
              strokeWidth={brush * 2}
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
          </React.Fragment>
        ))}
      </Svg>
      <View style={StyleSheet.absoluteFill} {...pan.panHandlers} />
      {all.length === 0 && (
        <View pointerEvents="none" style={styles.hint}>
          <View style={styles.hintGlyph}>
            <Svg width="34" height="34" viewBox="0 0 24 24">
              <Line x1="4" y1="20" x2="16" y2="6" stroke={theme.cyan} strokeWidth="1.8" strokeLinecap="round" />
              <Line x1="16" y1="6" x2="18" y2="16" stroke={theme.cyan} strokeWidth="1.8" strokeLinecap="round" />
              <Line x1="8" y1="6" x2="13" y2="4" stroke={theme.cyan} strokeWidth="1.8" strokeLinecap="round" />
            </Svg>
          </View>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    alignSelf: "center",
    backgroundColor: "#0a1029",
    borderWidth: 1,
    borderColor: "rgba(148,178,255,0.30)",
    overflow: "hidden",
  },
  svg: { backgroundColor: "transparent" },
  hint: { alignItems: "center", justifyContent: "center" },
  hintGlyph: { opacity: 0.5 },
});
