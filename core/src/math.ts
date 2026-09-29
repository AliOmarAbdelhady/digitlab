/** Numerically stable softmax (in place on a copy). */
export function softmax(scores: ArrayLike<number>): number[] {
  let max = -Infinity;
  for (let i = 0; i < scores.length; i++) if (scores[i] > max) max = scores[i];
  const out = new Array<number>(scores.length);
  let sum = 0;
  for (let i = 0; i < scores.length; i++) {
    out[i] = Math.exp(scores[i] - max);
    sum += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

export function argmax(xs: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < xs.length; i++) if (xs[i] > xs[best]) best = i;
  return best;
}
