#!/usr/bin/env bash
set -uo pipefail

# Unified CUDA differential probe runner.
#
# It deliberately uses the exact project build requested for CUDA validation.
# The detected capability is passed to the build so SM100/SM120-only NVFP4
# code is compiled only when the host can execute it.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

include_optin_gdn=0
while (($#)); do
  case "$1" in
    --include-optin-gdn)
      include_optin_gdn=1
      ;;
    -h|--help)
      cat <<'EOF'
Usage: tests/run_precision_suite.sh [--include-optin-gdn]

Runs the Candle, attention.rs, FlashInfer, FP8, GDN, and NVFP4 probes.
The SM90 persistent FlashInfer GQA probe is excluded by default because the
production path is opt-in. Pass --include-optin-gdn to audit it explicitly.
EOF
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi is required to detect CUDA compute capability" >&2
  exit 2
fi

mapfile -t raw_caps < <(
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
    | sed 's/[[:space:]]//g;/^$/d'
)
if ((${#raw_caps[@]} == 0)); then
  echo "error: nvidia-smi returned no compute capabilities" >&2
  exit 2
fi

caps=()
for raw in "${raw_caps[@]}"; do
  cap="$(awk -F. '{ printf "%d\n", ($1 * 10) + $2 }' <<<"$raw")"
  if [[ ! "$cap" =~ ^[0-9]+$ ]]; then
    echo "error: cannot parse compute capability '$raw'" >&2
    exit 2
  fi
  caps+=("$cap")
done

compute_cap="${caps[0]}"
for cap in "${caps[@]}"; do
  if ((cap < compute_cap)); then
    compute_cap="$cap"
  fi
done

if ((${#caps[@]} > 1)); then
  printf 'Detected GPU compute capabilities: %s; using lowest SM%s for all GPUs.\n' \
    "${raw_caps[*]}" "$compute_cap"
else
  printf 'Detected GPU compute capability: SM%s (%s)\n' "$compute_cap" "${raw_caps[0]}"
fi

export CUDA_COMPUTE_CAP="$compute_cap"
if ((compute_cap == 70 || compute_cap == 75)); then
  cuda_features="cuda,nccl"
  run_flashinfer=0
  # NVFP4 has a software CUDA implementation (decode, prefill/MoE, and
  # helpers) that must still be compared on legacy GPUs. Only the
  # Blackwell CUTLASS/FlashInfer NVFP4 cases are unavailable here.
  run_nvfp4=1
  if command -v nvcc >/dev/null 2>&1 && nvcc --version 2>/dev/null | grep -q 'release 13\.'; then
    echo "error: SM${compute_cap} requires an nvcc that supports compute_${compute_cap}; CUDA 13 removed this legacy target. Use CUDA 12.6 or another compatible CUDA 12 toolkit." >&2
    exit 2
  fi
  printf 'SM%s legacy CUDA path: using features %s; FlashInfer/CUTLASS hardware stages are skipped; software FP8/NVFP4 stages remain enabled.\n' \
    "$compute_cap" "$cuda_features"
else
  cuda_features="cuda,nccl,flashinfer,cutlass"
  run_flashinfer=1
  run_nvfp4=1
  printf 'SM%s CUDA path: using features %s.\n' "$compute_cap" "$cuda_features"
fi
if [[ -n "${RUSTFLAGS:-}" ]]; then
  export RUSTFLAGS="$RUSTFLAGS -C link-arg=-lstdc++"
else
  export RUSTFLAGS="-C link-arg=-lstdc++"
fi

out_dir="${XINFER_PRECISION_SUITE_DIR:-/tmp/xinfer_precision_suite_sm${compute_cap}_$$}"
mkdir -p "$out_dir/flashinfer"
mkdir -p "$out_dir/stages"
failed_case_log="$out_dir/failed_cases.txt"
: > "$failed_case_log"
echo "Probe outputs: $out_dir"

failures=0
run_stage() {
  local label="$1"
  shift
  local stage_log="$out_dir/stages/${label//[^[:alnum:]_.-]/_}.log"
  printf '\n===== %s =====\n' "$label"
  if "$@" 2>&1 | tee "$stage_log"; then
    echo "[OK] $label"
    return 0
  else
    local status=$?
    if ((status == 77)); then
      echo "[SKIP] $label (optional golden dependency unavailable)"
      return 0
    fi
  fi
  while IFS= read -r failed_line; do
    printf '%s: %s\n' "$label" "$failed_line" >> "$failed_case_log"
  done < <(grep -E '^[[:space:]]*FAIL[[:space:]]' "$stage_log" || true)
  echo "[FAIL] $label" >&2
  failures=$((failures + 1))
  return 1
}

if ! run_stage "required CUDA build and all Rust probes" \
  cargo build --release --features "$cuda_features" --examples; then
  echo "Build failed; probes were not run." >&2
  exit 1
fi

if run_stage "Candle precision probe" env \
  XINFER_CANDLE_PROBE="$out_dir/candle.bin" \
  "$ROOT_DIR/target/release/examples/candle_precision_probe"; then
  run_stage "Candle PyTorch/Q8_1 comparison" \
    python3 tests/probes/compare_candle_probe.py "$out_dir/candle.bin"
fi

if run_stage "attention.rs common-kernel probe" env \
  XINFER_MISC_PROBE="$out_dir/attention_misc.bin" \
  "$ROOT_DIR/target/release/examples/attention_misc_precision_probe"; then
  run_stage "attention.rs common-kernel PyTorch comparison" \
    python3 tests/probes/compare_attention_misc_probe.py "$out_dir/attention_misc.bin"
fi

if run_stage "FP8 probe" env \
  XINFER_FP8_PROBE="$out_dir/fp8.bin" \
  "$ROOT_DIR/target/release/examples/fp8_precision_probe"; then
  run_stage "FP8 PyTorch comparison" \
    python3 tests/probes/compare_fp8_probe.py "$out_dir/fp8.bin"
fi

gdn_probe_env=(XINFER_GDN_PROBE="$out_dir/gdn.bin")
if ((include_optin_gdn)); then
  gdn_probe_env+=(XINFER_GDN_PROBE_INCLUDE_FLASHINFER=1)
else
  gdn_probe_env+=(XINFER_GDN_PROBE_INCLUDE_FLASHINFER=0)
fi
if run_stage "GDN probe" env "${gdn_probe_env[@]}" \
  "$ROOT_DIR/target/release/examples/gdn_precision_probe"; then
  run_stage "GDN PyTorch comparison" \
    python3 tests/probes/compare_gdn_probe.py "$out_dir/gdn.bin"
fi

if ((run_flashinfer)); then
  if run_stage "FlashInfer paged-attention probe" env \
    XINFER_PROBE_DIR="$out_dir/flashinfer" \
    "$ROOT_DIR/target/release/examples/flashinfer_precision_probe"; then
    run_stage "FlashInfer/PyTorch/official-FlashInfer comparison" \
      python3 tests/probes/compare_flashinfer_probe.py "$out_dir/flashinfer"
  fi
else
  echo "SKIP FlashInfer stages: unavailable on SM${compute_cap} legacy feature set"
fi

if ((run_nvfp4)); then
  if run_stage "NVFP4 all-path probe (SM${compute_cap})" env \
    XINFER_NVFP4_PROBE="$out_dir/nvfp4.bin" \
    "$ROOT_DIR/target/release/examples/nvfp4_precision_probe"; then
    run_stage "NVFP4 independent PyTorch comparison" \
      python3 tests/probes/compare_nvfp4_probe.py "$out_dir/nvfp4.bin"
  fi
else
  echo "SKIP NVFP4 stages: unavailable on SM${compute_cap} legacy feature set"
fi

printf '\n===== suite result =====\n'
if ((failures == 0)); then
  echo "ALL AVAILABLE REQUESTED PROBES PASSED"
  echo "Results retained in: $out_dir"
  exit 0
fi
if [[ -s "$failed_case_log" ]]; then
  echo "FAILED CASES:"
  cat "$failed_case_log"
fi
echo "$failures probe stage(s) failed; results retained in: $out_dir" >&2
exit 1
