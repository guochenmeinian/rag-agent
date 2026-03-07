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
  ./run.sh test    # run src/tests/test_generator.py (full RAG E2E)
  ./run.sh test-milvus  # run src/tests/test_milvus.py
  ./run.sh doctor  # re-sign binary libs on macOS
  ./run.sh all     # setup + test

Environment variables:
  LLAMA_CLOUD_API_KEY   required for LlamaParse
  OPENAI_API_KEY        required for OpenAI answer generation
  RAG_MODEL_CACHE       optional, default: <repo>/.cache/modelscope
  RAG_TEST_QUERY        optional, default: 介绍一下技术规格
  RAG_DATA_DIR          optional, default: data
  RAG_OPENAI_MODEL      optional, default: gpt-4o-mini
  RAG_TOPK              optional, default: 5
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

  for pkg in PIL numpy pandas scipy; do
    local pkg_dir="$py_site/$pkg"
    if [[ -d "$pkg_dir" ]]; then
      find "$pkg_dir" -type f \( -name "*.so" -o -name "*.dylib" \) -print0 \
        | xargs -0 -I{} codesign --force --sign - "{}" >/dev/null 2>&1 || true
    fi
  done
}

validate_api_key() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "$value" ]]; then
    echo "ERROR: $name is not set."
    exit 1
  fi
  case "$value" in
    "你的key"|"your_key"|"YOUR_KEY")
      echo "ERROR: $name is placeholder text. Please set a real key."
      exit 1
      ;;
  esac
}

set_common_env() {
  ensure_venv
  export PYTHONPATH="$ROOT_DIR/src"
  export RAG_MODEL_CACHE="${RAG_MODEL_CACHE:-$ROOT_DIR/.cache/modelscope}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  mkdir -p "$RAG_MODEL_CACHE"
}

run_test_e2e() {
  set_common_env
  validate_api_key "LLAMA_CLOUD_API_KEY"
  validate_api_key "OPENAI_API_KEY"

  "$PYTHON_BIN" "$ROOT_DIR/src/tests/test_generator.py"
}

run_test_milvus() {
  set_common_env
  validate_api_key "LLAMA_CLOUD_API_KEY"
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
      run_test_e2e
      ;;
    test-milvus)
      run_test_milvus
      ;;
    doctor)
      ensure_venv
      fix_macos_signatures
      ;;
    all)
      install_deps
      fix_macos_signatures
      run_test_e2e
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
