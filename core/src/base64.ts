/**
 * Dependency-free base64 decoding + typed-array views.
 *
 * Hand-written instead of atob()/Buffer so the exact same code runs in browsers,
 * Node and React Native/Hermes. Produces a freshly allocated Uint8Array
 * (byteOffset 0), which is safe to view as Float32Array on little-endian
 * platforms (every browser, Node, Android and iOS device we target).
 */
const TABLE = new Int8Array(128).fill(-1);
const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
for (let i = 0; i < ALPHABET.length; i++) TABLE[ALPHABET.charCodeAt(i)] = i;

export function decodeBase64(input: string): Uint8Array {
  let n = 0;
  for (let i = 0; i < input.length; i++) {
    const c = input.charCodeAt(i);
    if (c === 61 /* '=' */) break;
    if (c < 128 && TABLE[c] >= 0) n++;
  }
  const out = new Uint8Array((n * 3) >> 2);
  let acc = 0;
  let bits = 0;
  let o = 0;
  for (let i = 0; i < input.length; i++) {
    const c = input.charCodeAt(i);
    if (c === 61) break;
    const v = c < 128 ? TABLE[c] : -1;
    if (v < 0) continue; // skip whitespace
    acc = (acc << 6) | v;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out[o++] = (acc >>> bits) & 0xff;
    }
  }
  return out;
}

/** Decode a base64 blob of little-endian float32s into a Float32Array view. */
export function base64ToF32(s: string): Float32Array {
  const u8 = decodeBase64(s);
  if (u8.length % 4 !== 0) throw new Error(`bad f32 blob length ${u8.length}`);
  return new Float32Array(u8.buffer, u8.byteOffset, u8.length / 4);
}

/** Decode a base64 blob of uint8s. */
export function base64ToU8(s: string): Uint8Array {
  return decodeBase64(s);
}
