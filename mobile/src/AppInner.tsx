import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  View,
  useWindowDimensions,
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import * as Haptics from "expo-haptics";
import { strokesToVector } from "../../core/src/preprocess";
import type { Prediction } from "../../core/src/models";
import DrawCanvas, { type Stroke } from "./DrawCanvas";
import ModelPicker from "./ModelPicker";
import Preview28 from "./Preview28";
import ResultPanel from "./ResultPanel";
import { MODEL_DEFS, useModels, type ModelKind } from "./useModels";
import { theme, FONTS } from "./theme";

export default function AppInner() {
  const { states, load, accuracy } = useModels();
  const [selected, setSelected] = useState<ModelKind>("cnn");
  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const [result, setResult] = useState<Prediction | null>(null);
  const [preview, setPreview] = useState<Uint8ClampedArray | null>(null);
  const [thinking, setThinking] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  const { width } = useWindowDimensions();
  const canvasSize = useMemo(
    () => Math.min(width - 48, 360),
    [width],
  );

  const def = MODEL_DEFS.find((d) => d.kind === selected)!;
  const hasInk = strokes.length > 0;

  useEffect(() => {
    load(selected).catch((e: Error) => setToast(`model load failed: ${e.message}`));
  }, [selected, load]);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 3200);
    return () => clearTimeout(t);
  }, [toast]);

  const onStrokeCommit = useCallback((s: Stroke) => {
    setStrokes((prev) => [...prev, s]);
    setResult(null);
  }, []);

  const clearAll = useCallback(() => {
    setStrokes([]);
    setResult(null);
    setPreview(null);
  }, []);

  const predictNow = useCallback(async () => {
    if (strokes.length === 0) {
      setToast("draw a digit first");
      return;
    }
    const pre = strokesToVector(strokes, canvasSize, canvasSize * 0.045);
    if (!pre) {
      setToast("ink not detected — draw a bigger digit");
      return;
    }
    setPreview(pre.image28);
    setThinking(true);
    try {
      const model = await load(selected);
      const r = model.predictFull(pre.vector);
      setResult(r);
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
    } catch (e: any) {
      setToast(`prediction failed: ${e?.message ?? e}`);
    } finally {
      setThinking(false);
    }
  }, [strokes, selected, load, canvasSize]);

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.scroll} bounces={false}>
        <View style={styles.header}>
          <View style={styles.logoMark}><Text style={styles.logoTxt}>7</Text></View>
          <View>
            <Text style={styles.logoName}>DigitLab</Text>
            <Text style={styles.logoSub}>MNIST · three models · one canvas</Text>
          </View>
        </View>

        <Text style={styles.kicker}>01 · CHOOSE YOUR MODEL</Text>
        <ModelPicker
          defs={MODEL_DEFS}
          selected={selected}
          states={states}
          accuracy={accuracy}
          onSelect={setSelected}
        />

        <Text style={styles.kicker}>02 · DRAW A DIGIT</Text>
        <DrawCanvas size={canvasSize} strokes={strokes} onStrokeCommit={onStrokeCommit} />

        <View style={styles.actions}>
          <Pressable onPress={clearAll} disabled={!hasInk} style={({ pressed }) => [
            styles.btnGhost, (!hasInk || pressed) && { opacity: !hasInk ? 0.4 : 0.8 },
          ]}>
            <Text style={styles.btnGhostTxt}>Clear</Text>
          </Pressable>
          <Pressable
            onPress={() => void predictNow()}
            disabled={!hasInk || thinking}
            style={({ pressed }) => [
              styles.btnPrimary,
              (!hasInk || thinking) && { opacity: 0.45 },
              pressed && hasInk && !thinking && { transform: [{ scale: 0.97 }] },
            ]}
          >
            <LinearGradient
              colors={["#22d3ee", "#6366f1", "#a855f7"]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={StyleSheet.absoluteFill}
            />
            <Text style={styles.btnPrimaryTxt}>
              {thinking ? "Predicting…" : `Predict · ${def.name}`}
            </Text>
          </Pressable>
        </View>

        <Text style={styles.kicker}>03 · PREDICTION</Text>
        <ResultPanel
          digit={result ? result.digit : null}
          probs={result ? result.probs : null}
          latencyMs={result ? result.latencyMs : null}
          modelName={def.name}
        />

        <View style={styles.previewRow}>
          <Preview28 image28={preview} />
        </View>

        <Text style={styles.footer}>
          trained on MNIST (60k) · all inference runs on-device
        </Text>
      </ScrollView>

      {toast && (
        <View style={styles.toastWrap}>
          <Text style={styles.toast}>{toast}</Text>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1 },
  scroll: { padding: 20, paddingBottom: 34, gap: 12 },
  header: { flexDirection: "row", alignItems: "center", gap: 13, marginBottom: 6 },
  logoMark: {
    width: 44, height: 44, borderRadius: 13,
    backgroundColor: "#6366f1",
    alignItems: "center", justifyContent: "center",
    shadowColor: "#3878ff", shadowRadius: 14, shadowOpacity: 0.45, shadowOffset: { width: 0, height: 4 },
    elevation: 6,
  },
  logoTxt: { fontFamily: FONTS.display, fontSize: 22, color: "#dff8ff" },
  logoName: { fontFamily: FONTS.display, fontSize: 20, color: theme.text },
  logoSub: { fontFamily: FONTS.regular, fontSize: 11.5, color: theme.dim, marginTop: 1 },
  kicker: {
    fontSize: 10, letterSpacing: 1.6, color: theme.faint,
    marginTop: 10, marginBottom: -2, fontFamily: FONTS.medium,
  },
  actions: { flexDirection: "row", gap: 12, justifyContent: "center", marginTop: 4 },
  btnGhost: {
    borderRadius: 14, borderWidth: 1, borderColor: theme.borderStrong,
    backgroundColor: theme.surface2, paddingVertical: 13, paddingHorizontal: 24,
  },
  btnGhostTxt: { color: theme.dim, fontFamily: FONTS.semiBold, fontSize: 14.5 },
  btnPrimary: {
    borderRadius: 14, overflow: "hidden", minWidth: 190,
    paddingVertical: 13, paddingHorizontal: 24,
    backgroundColor: "#3146c9",
    shadowColor: "#3870f6", shadowRadius: 16, shadowOpacity: 0.4, shadowOffset: { width: 0, height: 6 },
    elevation: 8,
    alignItems: "center",
  },
  btnPrimaryTxt: { color: "#04121f", fontFamily: FONTS.semiBold, fontSize: 14.5 },
  previewRow: { marginTop: 4 },
  footer: { textAlign: "center", color: theme.faint, fontSize: 11, marginTop: 14, fontFamily: FONTS.regular },
  toastWrap: {
    position: "absolute", bottom: 26, left: 0, right: 0,
    alignItems: "center",
  },
  toast: {
    color: theme.text, backgroundColor: "rgba(14,24,52,0.96)",
    borderWidth: 1, borderColor: theme.borderStrong,
    paddingHorizontal: 18, paddingVertical: 10, borderRadius: 13,
    fontSize: 13, fontFamily: FONTS.regular,
    overflow: "hidden",
  },
});
