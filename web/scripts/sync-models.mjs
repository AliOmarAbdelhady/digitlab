import { copyFileSync, mkdirSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const srcDir = path.join(root, "models");
const dstDir = path.join(root, "web", "public", "models");

mkdirSync(dstDir, { recursive: true });
for (const f of readdirSync(srcDir)) {
  if (/_mnist\.json$/.test(f)) {
    copyFileSync(path.join(srcDir, f), path.join(dstDir, f));
    console.log(`synced ${f}`);
  }
}
