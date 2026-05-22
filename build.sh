#!/usr/bin/env bash
set -euo pipefail

RELEASE=""
PROFILE="release"
FEATURES=""
PUBLISH=false

INSTALL=false
DST="/usr/local/bin"

usage() {
  cat <<EOF
Usage: $0 [--debug|--release] [--features "<feat1 feat2 ...>"] [publish] [--install] [--dst <dir>]

Options:
  --debug            Build debug profile
  --release          Build release profile (default)
  --features <...>   Cargo/Maturin feature list (string)
  publish            Publish to PyPI via maturin publish
  --install          Force --release build and copy xinfer into --dst (default: /usr/local/bin)
  --dst <dir>        Destination directory for --install (default: /usr/local/bin)
EOF
}

bin_name() {
  local name="$1"
  if [[ "${OSTYPE:-}" == "msys" || "${OSTYPE:-}" == "win32" || "${OSTYPE:-}" == "cygwin" ]]; then
    echo "${name}.exe"
  else
    echo "$name"
  fi
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --debug)
      RELEASE=""
      PROFILE="debug"
      ;;
    --release)
      RELEASE="--release"
      PROFILE="release"
      ;;
    --features)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --features requires a value"
        usage
        exit 1
      fi
      FEATURES="$2"
      shift
      ;;
    publish)
      PUBLISH=true
      ;;
    --install)
      INSTALL=true
      ;;
    --dst)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --dst requires a value"
        usage
        exit 1
      fi
      DST="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

if [[ "$INSTALL" == true ]]; then
  RELEASE="--release"
  PROFILE="release"
fi

XINFER_BIN="$(bin_name xinfer)"

HAS_PYTHON=false
if [[ "$FEATURES" == *"python"* ]]; then
  HAS_PYTHON=true
fi

IS_METAL=false
if [[ "$FEATURES" == *"metal"* ]]; then
  IS_METAL=true
fi

echo "Building with profile: $PROFILE"
echo "Features: $FEATURES"
echo "Install: $INSTALL"

# -------------------------------------------------------------------
# INSTALL PATH: build xinfer binary, install to --dst
# -------------------------------------------------------------------
if [[ "$INSTALL" == true ]]; then
  echo "Binary-only install requested; skipping maturin and python package staging."

  FEATURES_NO_PY=$(echo "$FEATURES" | sed -E 's/\bpython\b//g' | xargs)
  echo "Building xinfer binary..."
  cargo build $RELEASE --bin xinfer --features "$FEATURES_NO_PY"

  echo "Installing binary to: $DST"
  mkdir -p "$DST"

  XINFER_PATH="target/$PROFILE/$XINFER_BIN"
  if [[ ! -f "$XINFER_PATH" ]]; then
    echo "Error: xinfer binary not found at $XINFER_PATH"
    exit 1
  fi
  install -m 755 "$XINFER_PATH" "$DST/xinfer"

  echo "Build and install complete."
  exit 0
fi

# -------------------------------------------------------------------
# NO PYTHON: build xinfer binary, done
# -------------------------------------------------------------------
if [[ "$HAS_PYTHON" != true ]]; then
  echo "Building xinfer binary..."
  cargo build $RELEASE --bin xinfer --features "$FEATURES"
  echo "Build complete."
  exit 0
fi

# -------------------------------------------------------------------
# PYTHON PATH: python package staging + maturin build/publish
# -------------------------------------------------------------------
DEST_DIR="xinfer"
mkdir -p "$DEST_DIR"

if [[ "$IS_METAL" == true ]]; then
  echo "Metal feature detected. Skipping xinfer binary copy for python package."
else
  FEATURES_BIN=$(echo "$FEATURES" | sed -E 's/\bpython\b//g' | xargs)
  echo "Building xinfer binary..."
  cargo build $RELEASE --bin xinfer --features "$FEATURES_BIN"

  echo "Copying xinfer binary into $DEST_DIR/ ..."
  XINFER_BINARY="target/$PROFILE/$XINFER_BIN"
  cp "$XINFER_BINARY" "$DEST_DIR/xinfer"
  chmod 755 "$DEST_DIR/xinfer"
  if command -v patchelf >/dev/null 2>&1; then
    echo "Patching xinfer rpath for bundled libs..."
    patchelf --set-rpath '$ORIGIN:$ORIGIN/../xinfer.libs' "$DEST_DIR/xinfer"
  else
    echo "Warning: patchelf not found; xinfer may need LD_LIBRARY_PATH to find bundled libs."
  fi
fi

cp "xinfer.pyi" "$DEST_DIR/__init__.pyi"
chmod 755 "$DEST_DIR/__init__.pyi"
touch "$DEST_DIR/py.typed"
chmod 755 "$DEST_DIR/py.typed"
cp "python/__init__.py" "$DEST_DIR/__init__.py"
chmod 755 "$DEST_DIR/__init__.py"
cp "ReadMe.md" "$DEST_DIR/ReadMe.md"
chmod 755 "$DEST_DIR/ReadMe.md"
cp "example/server.py" "$DEST_DIR/server.py"
chmod 755 "$DEST_DIR/server.py"
cp "example/chat.py" "$DEST_DIR/chat.py"
chmod 755 "$DEST_DIR/chat.py"
cp "example/completion.py" "$DEST_DIR/completion.py"
chmod 755 "$DEST_DIR/completion.py"

echo "Building Python extension with maturin..."

FEATURES_MATURIN=$(echo "$FEATURES" | sed -E 's/\bflashattn\b//g' | xargs)
FEATURES_MATURIN=$(echo "$FEATURES_MATURIN" | sed -E 's/\bflashinfer\b//g' | xargs)

echo "Python extension features: $FEATURES_MATURIN"

if [[ "$PUBLISH" == true ]]; then
  echo "Publishing package to PyPI..."
  maturin publish --features "$FEATURES_MATURIN" --username __token__
else
  maturin build $RELEASE --features "$FEATURES_MATURIN"
fi

echo "Cleaning up temporary files..."
if [[ "$IS_METAL" != true ]]; then
  rm -f "$DEST_DIR/xinfer"
fi
rm -f "$DEST_DIR/__init__.py" \
      "$DEST_DIR/__init__.pyi" \
      "$DEST_DIR/py.typed" \
      "$DEST_DIR/ReadMe.md" \
      "$DEST_DIR/server.py" \
      "$DEST_DIR/chat.py" \
      "$DEST_DIR/completion.py"
rm -rf "$DEST_DIR"

echo "Build complete."
