#!/usr/bin/env bash
# llm-for-the-poors — install script.
#
# What this sets up:
#   1. llama.cpp prebuilt server (Vulkan, x64)  -> llama/llama-native/
#      (fallback: builds llama-cpp-python in .venv/)
#   2. python venv + huggingface_hub             -> .venv/
#   3. kompact context-compression proxy (venv)  -> ~/kompact/
#   4. HF_TOKEN/HF_HUB_DISABLE_XET in your shell rc (you paste the token)
#
# Models are NOT downloaded here — pick one on first run (see README).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say(){ printf '\033[32;1m==> %s\033[0m\n' "$*"; }
warn(){ printf '\033[33;1m!! %s\033[0m\n' "$*"; }
die(){ printf '\033[31;1mError:\033[0m %s\n' "$*" >&2; exit 1; }

command -v python3 >/dev/null || die "python3 required: sudo apt install -y python3 python3-venv python3-pip"
command -v curl    >/dev/null || die "curl required:  sudo apt install -y curl"
command -v tar     >/dev/null || die "tar required"

# Vulkan runtime lib (AMD/Intel driver already ships the userspace layer)
if ! ldconfig -p 2>/dev/null | grep -q "libvulkan.so"; then
  warn "libvulkan not found. Install a Vulkan driver, e.g.:
         sudo apt install -y libvulkan1 mesa-vulkan-drivers"
fi

# 1. llama.cpp prebuilt (Vulkan, x64) — no compilers, no build deps
NATIVE="$HERE/llama/llama-native"
if [ ! -x "$NATIVE/llama-server" ]; then
  say "downloading llama.cpp prebuilt (ubuntu-vulkan-x64)..."
  B=$(curl -fsSL https://api.github.com/repos/ggml-org/llama.cpp/releases/latest \
        | grep -oE '"tag_name": *"[^"]+"' | head -1 | grep -oE 'b[0-9]+' || true)
  [ -n "$B" ] || die "could not resolve latest llama.cpp build"
  URL="https://github.com/ggml-org/llama.cpp/releases/download/$B/llama-$B-bin-ubuntu-vulkan-x64.tar.gz"
  curl -fL --retry 3 -o /tmp/llama-native.tar.gz "$URL"
  rm -rf /tmp/llama-dl && mkdir -p /tmp/llama-dl
  tar -xzf /tmp/llama-native.tar.gz -C /tmp/llama-dl
  SVR="$(find /tmp/llama-dl -name llama-server -type f | head -1)"
  [ -n "$SVR" ] || die "no llama-server found inside the prebuilt tarball"
  mkdir -p "$NATIVE"
  cp -r "$(dirname "$SVR")"/. "$NATIVE"/
  chmod +x "$NATIVE/llama-server"
  say "native llama-server -> $NATIVE"
else
  say "llama-server already present ($NATIVE)"
fi

# 2. python venv (llama-cpp-python fallback backend; also used by the proxy)
if [ ! -x "$HERE/.venv/bin/python" ]; then
  say "creating .venv..."
  python3 -m venv "$HERE/.venv"
fi
"$HERE/.venv/bin/pip" install -q --upgrade pip
"$HERE/.venv/bin/pip" install -q huggingface_hub >/dev/null 2>&1 || true
say "venv ready -> $HERE/.venv"

# 3. kompact — context compression proxy (optional but recommended)
KOMPACT_BIN="${KOMPACT:-$HOME/kompact/bin/kompact}"
if [ ! -x "$KOMPACT_BIN" ]; then
  say "installing kompact in ~/kompact (40-70% context savings)..."
  python3 -m venv "$HOME/kompact"
  "$HOME/kompact/bin/pip" install -q --upgrade pip
  "$HOME/kompact/bin/pip" install -q kompact
else
  say "kompact already present ($KOMPACT_BIN)"
fi

# 4. Hugging Face token (free) for model downloads + Xet CDN workaround
if grep -qs "^export HF_TOKEN=" "$HOME/.bashrc" "$HOME/.zshrc" 2>/dev/null; then
  say "HF_TOKEN already set in your shell rc."
else
  warn "Models are pulled from Hugging Face; you need a free read token."
  printf 'Paste your token (https://huggingface.co/settings/tokens) or Enter to skip: '
  read -rs token; printf '\n'
  if [ -n "$token" ]; then
    RC="$HOME/.bashrc"; [ -n "${ZSH_VERSION:-}" ] && RC="$HOME/.zshrc"
    {
      printf '\nexport HF_TOKEN=%s\nexport HF_HUB_DISABLE_XET=1\n' "$token"
    } >> "$RC"
    warn "token written to $RC (source it in new shells)"
  else
    warn "skipped — run llm.sh start with HF_TOKEN set via ENV."
  fi
fi

say "install complete."
say " next:  ./manage_deepseek.py install --quant Q2_5_C7B   # Qwen-Coder-7B (4.7GB) or your pick"
say "        ./llm.sh start                                  # starts the full chain on :8000"
say "        curl http://127.0.0.1:8000/health               # sanity check"
say " full model list + opencode setup: see README.md"