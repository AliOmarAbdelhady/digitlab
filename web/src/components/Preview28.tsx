import { useEffect, useRef } from "react";

interface Props {
  image28: Uint8ClampedArray | null;
}

/** Tiny 28×28 preview — "what the model sees" after MNIST preprocessing. */
export default function Preview28({ image28 }: Props) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, 28, 28);
    if (!image28) return;
    const img = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = image28[i];
      img.data[i * 4] = Math.round(v * 0.72 + 20); // slight blue tint
      img.data[i * 4 + 1] = Math.round(v * 0.85 + 30);
      img.data[i * 4 + 2] = Math.round(v + 60);
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [image28]);

  return (
    <div className="preview-panel">
      <div className="preview-title">what the model sees</div>
      <canvas ref={ref} width={28} height={28} className="preview28" />
      <div className="preview-note">crop → 20px box → center of mass</div>
    </div>
  );
}
