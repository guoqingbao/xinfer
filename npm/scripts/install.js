#!/usr/bin/env node
"use strict";

const fs = require("fs");
const https = require("https");
const os = require("os");
const path = require("path");
const { createHash } = require("crypto");
const { spawnSync } = require("child_process");

const pkg = require("../package.json");

function rmrf(target) {
  try {
    if (fs.rmSync) {
      fs.rmSync(target, { recursive: true, force: true });
    } else if (fs.rmdirSync) {
      fs.rmdirSync(target, { recursive: true });
    }
  } catch (_) {}
}

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

/**
 * Detect CUDA compute capability via nvidia-smi.
 * Returns the highest SM version across all GPUs (e.g. "90", "89", "80").
 * Returns null if no NVIDIA GPU detected (e.g. macOS Metal).
 */
function detectCudaComputeCap() {
  // Allow explicit override
  if (process.env.XINFER_CUDA_COMPUTE_CAP) {
    return process.env.XINFER_CUDA_COMPUTE_CAP;
  }

  try {
    const result = spawnSync(
      "nvidia-smi",
      [
        "--query-gpu=compute_cap",
        "--format=csv,noheader,nounits",
      ],
      { encoding: "utf8", timeout: 10000 }
    );
    if (result.status !== 0 || !result.stdout) return null;

    let maxCap = 0;
    for (const line of result.stdout.trim().split("\n")) {
      const cap = line.trim();
      if (!cap) continue;
      // Parse "8.9" -> 89, "9.0" -> 90, "10.0" -> 100
      const parts = cap.split(".");
      const major = parseInt(parts[0], 10);
      const minor = parseInt(parts[1] || "0", 10);
      const sm = major * 10 + minor;
      if (sm > maxCap) maxCap = sm;
    }
    return maxCap > 0 ? String(maxCap) : null;
  } catch {
    return null;
  }
}

/**
 * Map detected SM version to the build variant name.
 * Build variants match CI matrix: sm70, sm75, sm80, sm89, sm90, sm120
 */
function smToVariant(sm) {
  const n = parseInt(sm, 10);
  if (n >= 120) return "sm120";
  if (n >= 90) return "sm90";
  if (n >= 89) return "sm89";
  if (n >= 80) return "sm80";
  if (n >= 75) return "sm75";
  if (n >= 70) return "sm70";
  return "sm80";
}

function download(url, destination, redirects = 0) {
  if (redirects > 5) {
    return Promise.reject(
      new Error(`Too many redirects while downloading ${url}`)
    );
  }
  return new Promise((resolve, reject) => {
    const proto = url.startsWith("https") ? https : require("http");
    const request = proto.get(url, (response) => {
      if (
        response.statusCode >= 300 &&
        response.statusCode < 400 &&
        response.headers.location
      ) {
        response.resume();
        const next = new URL(response.headers.location, url).toString();
        download(next, destination, redirects + 1).then(resolve, reject);
        return;
      }
      if (response.statusCode !== 200) {
        response.resume();
        reject(
          new Error(`Download failed (${response.statusCode}) for ${url}`)
        );
        return;
      }
      const file = fs.createWriteStream(destination);
      response.pipe(file);
      file.on("finish", () => file.close(resolve));
      file.on("error", reject);
    });
    request.on("error", reject);
  });
}

function sha256(file) {
  const hash = createHash("sha256");
  hash.update(fs.readFileSync(file));
  return hash.digest("hex");
}

function expectedSha(manifest, artifactName) {
  for (const line of manifest.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const parts = trimmed.split(/\s+/);
    const checksum = parts[0];
    const name = parts[parts.length - 1].replace(/^\*/, "").replace(/^\.\//, "");
    if (name === artifactName) return checksum;
  }
  return null;
}

async function main() {
  const target = targetTriple();
  const version = process.env.XINFER_INSTALL_VERSION || pkg.assetVersion || pkg.version;
  const tag = process.env.XINFER_INSTALL_TAG || `v${version}`;
  const base =
    process.env.XINFER_INSTALL_BASE_URL ||
    `https://github.com/guoqingbao/xinfer/releases/download/${tag}`;

  let variant = "";
  if (process.platform === "linux") {
    const sm = detectCudaComputeCap();
    if (sm) {
      variant = `-${smToVariant(sm)}`;
      console.log(
        `Detected CUDA compute capability: sm_${sm} -> variant${variant}`
      );
    } else {
      console.log(
        "No NVIDIA GPU detected; downloading default (sm80) build."
      );
      variant = "-sm80";
    }
  }
  // macOS uses metal variant
  if (process.platform === "darwin") {
    variant = "-metal";
  }

  const artifactName = `xinfer-${version}-${target}${variant}.tar.gz`;
  const artifactUrl = `${base}/${artifactName}`;
  const sumsUrl = `${base}/SHA256SUMS`;
  const root = path.resolve(__dirname, "..");
  const installDir = path.join(root, "vendor", `${target}${variant}`);
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "xinfer-install-"));
  const archivePath = path.join(tmpDir, artifactName);
  const sumsPath = path.join(tmpDir, "SHA256SUMS");

  try {
    console.log(`Downloading ${artifactUrl}...`);
    await download(sumsUrl, sumsPath);
    await download(artifactUrl, archivePath);
    const expected = expectedSha(
      fs.readFileSync(sumsPath, "utf8"),
      artifactName
    );
    if (!expected) {
      throw new Error(`No checksum entry found for ${artifactName}`);
    }
    const actual = sha256(archivePath);
    if (actual !== expected) {
      throw new Error(
        `Checksum mismatch for ${artifactName}: expected ${expected}, got ${actual}`
      );
    }

    rmrf(installDir);
    fs.mkdirSync(installDir, { recursive: true });
    const tar = spawnSync(
      "tar",
      ["-xzf", archivePath, "-C", installDir],
      { stdio: "inherit" }
    );
    if (tar.error) throw tar.error;
    if (tar.status !== 0) {
      throw new Error(`tar exited with status ${tar.status}`);
    }
    const binary = path.join(installDir, "xinfer");
    fs.chmodSync(binary, 0o755);
    const hint = [
      `xinfer installed to ${installDir}`,
      "",
      "============================================",
      " xInfer binary installed via npm",
      "============================================",
      "",
      " HuggingFace model:",
      "   xinfer --m Qwen/Qwen3-8B --ui-server",
      "",
      " Local model path:",
      "   xinfer --w /path/to/model --ui-server",
      "",
      " API server (no UI):",
      "   xinfer --m Qwen/Qwen3-8B --server",
      "============================================",
    ].join("\n");
    process.stderr.write(hint + "\n");
  } finally {
    rmrf(tmpDir);
  }
}

main().catch((err) => {
  console.error(`xinfer install failed: ${err.message}`);
  process.exit(1);
});
