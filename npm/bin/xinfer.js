#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

function targetTriple() {
  const platform = process.platform;
  const arch = process.arch;
  if (
    (platform !== "linux" && platform !== "darwin") ||
    (arch !== "x64" && arch !== "arm64")
  ) {
    throw new Error(
      `Unsupported platform: ${platform}-${arch}. Supported: linux/darwin x64/arm64.`
    );
  }
  return `${platform}-${arch}`;
}

function findBinary() {
  const target = targetTriple();
  const root = path.resolve(__dirname, "..");

  // Prefer arch-specific variant installed by postinstall
  const variants = fs.readdirSync(path.join(root, "vendor")).filter((d) =>
    d.startsWith(target)
  );
  if (variants.length > 0) {
    variants.sort();
    const best = variants[variants.length - 1];
    const bin = path.join(root, "vendor", best, "xinfer");
    if (fs.existsSync(bin)) return bin;
  }

  // Fallback to plain target
  const fallback = path.join(root, "vendor", target, "xinfer");
  if (fs.existsSync(fallback)) return fallback;

  throw new Error(
    `xinfer native binary is missing for ${target}. Reinstall the package or set XINFER_INSTALL_BASE_URL for a custom release mirror.`
  );
}

function main() {
  const binary = findBinary();
  const result = spawnSync(binary, process.argv.slice(2), {
    stdio: "inherit",
    env: process.env,
  });
  if (result.error) {
    throw result.error;
  }
  if (result.signal) {
    process.kill(process.pid, result.signal);
  }
  process.exit(result.status === null ? 1 : result.status);
}

try {
  main();
} catch (err) {
  console.error(`xinfer: ${err.message}`);
  process.exit(1);
}
