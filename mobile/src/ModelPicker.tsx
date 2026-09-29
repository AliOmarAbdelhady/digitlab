import React from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { theme, FONTS } from "./theme";
import type { LoadState, ModelDef, ModelKind } from "./useModels";

interface Props {
  defs: ModelDef[];
  selected: ModelKind;
  states: Record<ModelKind, LoadState>;
  accuracy: (k: ModelKind) => number | null;
  onSelect: (k: ModelKind) => void;
}

export default function ModelPicker({ defs, selected, states, accuracy, onSelect }: Props) {
  return (
    <View style={styles.row}>
      {defs.map((d) => {
        const state = states[d.kind];
        const acc = accuracy(d.kind);
        const isSel = selected === d.kind;
        return (
          <Pressable
            key={d.kind}
            onPress={() => onSelect(d.kind)}
            style={[styles.card, isSel && styles.cardSel]}
          >
            <View style={styles.top}>
              <Text style={[styles.name, { color: isSel ? "#c9f0fa" : theme.text }]}>{d.name}</Text>
              {d.recommended ? <View style={styles.badge}><Text style={styles.badgeTxt}>BEST</Text></View> : null}
            </View>
            <Text style={styles.tag}>{d.tag}</Text>
            <Text style={styles.desc} numberOfLines={1}>
              {acc !== null ? `${(acc * 100).toFixed(2)}% test` : d.desc}
            </Text>
            <View style={[styles.dot, state === "ready" && styles.dotReady, state === "loading" && styles.dotLoad]} />
          </Pressable>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: "row", gap: 10 },
  card: {
    flex: 1,
    borderRadius: 16,
    padding: 12,
    backgroundColor: theme.surface,
    borderWidth: 1,
    borderColor: theme.border,
    minHeight: 92,
  },
  cardSel: {
    backgroundColor: "rgba(34,211,238,0.10)",
    borderColor: "rgba(94,200,250,0.65)",
  },
  top: { flexDirection: "row", alignItems: "center", gap: 6 },
  name: { fontFamily: FONTS.display, fontSize: 18, color: theme.text },
  badge: {
    backgroundColor: "#22d3ee",
    borderRadius: 999,
    paddingHorizontal: 6,
    paddingVertical: 1,
  },
  badgeTxt: { color: "#06202b", fontSize: 8.5, fontFamily: FONTS.semiBold, letterSpacing: 0.5 },
  tag: { color: theme.cyan, fontSize: 10.5, marginTop: 2, fontFamily: FONTS.medium },
  desc: { color: theme.faint, fontSize: 10.5, marginTop: 4, fontFamily: FONTS.regular },
  dot: {
    position: "absolute", right: 10, top: 10,
    width: 7, height: 7, borderRadius: 4,
    backgroundColor: "rgba(148,178,255,0.25)",
  },
  dotReady: { backgroundColor: theme.mint },
  dotLoad: { backgroundColor: theme.cyan },
});
