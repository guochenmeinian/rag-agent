#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"

usage() {
  cat <<'EOF'
Usage:
  ./run.sh setup   # create .venv and install pinned deps
  ./run.sh test    # run src/tests/test_milvus.py
  ./run.sh all     # setup + test

Environment variables:
  LLAMA_CLOUD_API_KEY   required for LlamaParse
  RAG_MODEL_CACHE       optional, default: <repo>/.cache/modelscope
  VENV_DIR              optional, default: <repo>/.venv
EOF
}

ensure_venv() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    python3 -m venv "$VENV_DIR"
  fi
}

install_deps() {
  ensure_venv
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PIP_BIN" install -r "$ROOT_DIR/requirements.txt"
}

fix_macos_signatures() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    return 0
  fi
  if ! command -v codesign >/dev/null 2>&1; then
    return 0
  fi

  local py_site
  py_site="$("$PYTHON_BIN" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"

  for pkg in PIL numpy; do
    local pkg_dir="$py_site/$pkg"
    if [[ -d "$pkg_dir" ]]; then
      find "$pkg_dir" -type f \( -name "*.so" -o -name "*.dylib" \) -print0 \
        | xargs -0 -I{} codesign --force --sign - "{}" >/dev/null 2>&1 || true
    fi
  done
}

run_test() {
  ensure_venv
  export PYTHONPATH="$ROOT_DIR/src"
  export RAG_MODEL_CACHE="${RAG_MODEL_CACHE:-$ROOT_DIR/.cache/modelscope}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

  if [[ -z "${LLAMA_CLOUD_API_KEY:-}" ]]; then
    echo "ERROR: LLAMA_CLOUD_API_KEY is not set."
    echo "Example: export LLAMA_CLOUD_API_KEY='your_real_key'"
    exit 1
  fi

  mkdir -p "$RAG_MODEL_CACHE"
  "$PYTHON_BIN" "$ROOT_DIR/src/tests/test_milvus.py"
}

main() {
  local cmd="${1:-all}"
  case "$cmd" in
    setup)
      install_deps
      fix_macos_signatures
      ;;
    test)
      run_test
      ;;
    all)
      install_deps
      fix_macos_signatures
      run_test
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
