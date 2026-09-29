import React, { useMemo } from "react";
import { StyleSheet, Text, View } from "react-native";
import Svg, { Rect } from "react-native-svg";
import { theme, FONTS } from "./theme";

/** 28×28 "what the model sees" preview, rendered as merged-run SVG rects. */
export default function Preview28({ image28 }: { image28: Uint8ClampedArray | null }) {
  const runs = useMemo(() => {
    if (!image28) return [];
    const out: { x: number; y: number; w: number; a: number }[] = [];
    for (let y = 0; y < 28; y++) {
      let x = 0;
      while (x < 28) {
        const v = image28[y * 28 + x];
        if (v > 8) {
          let x2 = x;
          let vmax = v;
          while (x2 + 1 < 28 && image28[y * 28 + x2 + 1] > 8) {
            x2++;
            vmax = Math.max(vmax, image28[y * 28 + x2]);
          }
          out.push({ x, y, w: x2 - x + 1, a: vmax / 255 });
          x = x2 + 1;
        } else x++;
      }
    }
    return out;
  }, [image28]);

  return (
    <View style={styles.wrap}>
      <Text style={styles.title}>WHAT THE MODEL SEES</Text>
      <View style={styles.frame}>
        <Svg width="100%" height="100%" viewBox="0 0 28 28">
          <Rect x="0" y="0" width="28" height="28" fill="#050914" />
          {runs.map((r, i) => (
            <Rect key={i} x={r.x} y={r.y} width={r.w} height={1} fill="#63b3ff" opacity={r.a} />
          ))}
        </Svg>
      </View>
      <Text style={styles.note}>crop → 20px box → center of mass</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { flexDirection: "row", alignItems: "center", gap: 12 },
  title: { fontSize: 9, letterSpacing: 1.4, color: theme.faint, flex: 1, fontFamily: FONTS.medium },
  frame: {
    width: 86, height: 86, borderRadius: 10,
    borderWidth: 1, borderColor: theme.border, overflow: "hidden",
  },
  note: { fontSize: 9, color: theme.faint, flex: 1, textAlign: "right", fontFamily: FONTS.regular },
});
