import React from "react";
import { StyleSheet, Text, View } from "react-native";
import Svg, { Circle, Defs, LinearGradient, Stop } from "react-native-svg";
import { theme, FONTS } from "./theme";

interface Props {
  digit: number | null;
  probs: number[] | null;
  latencyMs: number | null;
  modelName: string;
}

function Ring({ value }: { value: number }) {
  const R = 44;
  const C = 2 * Math.PI * R;
  return (
    <View style={styles.ringWrap}>
      <Svg width="104" height="104" style={{ transform: [{ rotate: "-90deg" }] }}>
        <Circle cx="52" cy="52" r={R} stroke="rgba(148,178,255,0.14)" strokeWidth="8" fill="none" />
        <Circle
          cx="52" cy="52" r={R}
          stroke="url(#g)" strokeWidth="8" fill="none"
          strokeDasharray={C}
          strokeDashoffset={C * (1 - value)}
          strokeLinecap="round"
        />
        <Defs>
          <LinearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
            <Stop offset="0%" stopColor="#22d3ee" />
            <Stop offset="100%" stopColor="#6366f1" />
          </LinearGradient>
        </Defs>
      </Svg>
      <View style={styles.ringLabel}>
        <Text style={styles.ringPct}>{(value * 100).toFixed(1)}%</Text>
        <Text style={styles.ringSub}>confidence</Text>
      </View>
    </View>
  );
}

export default function ResultPanel({ digit, probs, latencyMs, modelName }: Props) {
  if (digit === null || !probs) {
    return (
      <View style={[styles.panel, styles.empty]}>
        <Text style={styles.ghost}>?</Text>
        <Text style={styles.emptyText}>draw a digit, then press Predict</Text>
      </View>
    );
  }
  const top3 = probs
    .map((p, i) => ({ d: i, p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 3);

  return (
    <View style={styles.panel}>
      <View style={styles.left}>
        <Text style={styles.digit}>{digit}</Text>
        <View style={styles.chip}><Text style={styles.chipTxt}>{modelName}</Text></View>
      </View>
      <Ring value={probs[digit] ?? 0} />
      <View style={styles.bars}>
        {top3.map((t, i) => (
          <View key={t.d} style={styles.barRow}>
            <Text style={styles.barDigit}>{t.d}</Text>
            <View style={styles.barTrack}>
              <View style={[styles.barFill, i === 0 && styles.barFirst, { width: `${Math.max(3, t.p * 100)}%` }]} />
            </View>
          </View>
        ))}
        {latencyMs !== null && (
          <Text style={styles.latency}>⚡ {latencyMs.toFixed(0)} ms · on-device</Text>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  panel: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    borderRadius: 20,
    backgroundColor: theme.surface,
    borderWidth: 1,
    borderColor: theme.border,
    padding: 14,
    minHeight: 132,
  },
  empty: { justifyContent: "center", gap: 6 },
  ghost: { fontFamily: FONTS.display, fontSize: 64, color: "rgba(148,178,255,0.16)" },
  emptyText: { color: theme.dim, fontSize: 12.5, fontFamily: FONTS.regular },
  left: { alignItems: "center", gap: 8 },
  digit: {
    fontFamily: FONTS.display,
    fontSize: 76,
    color: "#8ee9f7",
    textShadowColor: "rgba(56,130,246,0.5)",
    textShadowRadius: 22,
  },
  chip: {
    borderWidth: 1,
    borderColor: "rgba(34,211,238,0.35)",
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 3,
    backgroundColor: "rgba(34,211,238,0.08)",
  },
  chipTxt: { color: theme.cyan, fontSize: 9.5, letterSpacing: 1.2, fontFamily: FONTS.semiBold },
  ringWrap: { width: 104, height: 104 },
  ringLabel: { position: "absolute", left: 0, right: 0, top: 0, bottom: 0, alignItems: "center", justifyContent: "center" },
  ringPct: { fontFamily: FONTS.displayMed, fontSize: 17, color: theme.text },
  ringSub: { fontSize: 8, letterSpacing: 1, color: theme.faint, marginTop: 1 },
  bars: { flex: 1, gap: 7, marginLeft: 6 },
  barRow: { flexDirection: "row", alignItems: "center", gap: 7 },
  barDigit: { fontFamily: FONTS.displayMed, fontSize: 13, color: theme.dim, width: 14, textAlign: "center" },
  barTrack: { flex: 1, height: 8, borderRadius: 999, backgroundColor: "rgba(148,178,255,0.10)", overflow: "hidden" },
  barFill: { height: "100%", borderRadius: 999, backgroundColor: "rgba(99,102,241,0.6)" },
  barFirst: { backgroundColor: "#22d3ee" },
  latency: { color: theme.faint, fontSize: 10, marginTop: 2, fontFamily: FONTS.regular },
});
