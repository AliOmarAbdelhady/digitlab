import { copyFileSync, mkdirSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const srcDir = path.join(root, "models");
const dstDir = path.join(root, "mobile", "assets", "models");

mkdirSync(dstDir, { recursive: true });
for (const f of readdirSync(srcDir)) {
  if (/_mnist\.json$/.test(f)) {
    const dst = path.join(dstDir, f.replace(/\.json$/, ".dlmodel"));
    copyFileSync(path.join(srcDir, f), dst);
    console.log(`synced ${f} -> ${path.basename(dst)}`);
  }
}
