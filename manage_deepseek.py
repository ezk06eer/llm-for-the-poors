#!/usr/bin/env python3
"""Manages a local DeepSeek-Coder-V2-Lite-Instruct GGUF on this machine.

  python3 manage_deepseek.py status
  python3 manage_deepseek.py install            # setup runtime + download model
  python3 manage_deepseek.py chat               # interactive session on GPU
  python3 manage_deepseek.py verify             # smoke test + tokens/sec
  python3 manage_deepseek.py serve              # OpenAI-compatible API server
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(BASE, ".venv")
BIN = os.path.join(VENV, "bin")
PY = os.path.join(BIN, "python")
BACKEND_FILE = os.path.join(BASE, ".backend")
QUANT_FILE = os.path.join(BASE, ".quant")
REPO = "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF"
FILES = {
    "Q3_K_M": (REPO, "DeepSeek-Coder-V2-Lite-Instruct-Q3_K_M.gguf", 8.1e9),
    "Q4_K_M": (REPO, "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf", 10.4e9),
    "Q2_5_C7B": ("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF", "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf", 4.68e9),
    "HE8B": ("bartowski/Hermes-2-Pro-Llama-3-8B-GGUF", "Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf", 4.92e9),
    "ML8B": ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", 4.92e9),
    "ABL8B": ("bartowski/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF", "Meta-Llama-3.1-8B-Instruct-abliterated-Q4_K_M.gguf", 4.92e9),
}
APT = "libvulkan-dev glslang-tools glslc spirv-headers"

WARN = "\033[33m{}\033[0m"
OK = "\033[32m{}\033[0m"
ERR = "\033[31m{}\033[0m"


def sh(args, env=None, check=False):
    return subprocess.run(
        args, env=env, text=True, check=check,
        stdout=subprocess.DEVNULL if args[0].endswith("pip") else None,
    )


def ensure_venv():
    if not os.path.exists(PY):
        print("creating venv...")
        subprocess.run([sys.executable, "-m", "venv", VENV], check=True)
    subprocess.run([os.path.join(BIN, "pip"), "install", "-q", "--upgrade", "pip"], check=False)


def vulkan_sdk_ready():
    return (
        os.path.exists("/usr/include/vulkan/vulkan.h")
        and shutil.which("glslangValidator") is not None
        and shutil.which("glslc") is not None
        and os.path.exists("/usr/share/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake")
    )


def install_sdk():
    if vulkan_sdk_ready():
        return
    sudo = [] if os.geteuid() == 0 else ["sudo", "-n"]
    print(f"[sudo] installing Vulkan build deps: {APT}")
    r = sh([*sudo, "apt-get", "update"])
    if r.returncode != 0:
        print(ERR.format("apt update failed. Run this once in your terminal, then rerun:"))
        print(ERR.format("    sudo apt-get update && sudo apt-get install -y " + APT))
        sys.exit(1)
    r = sh([*sudo, "apt-get", "install", "-y", *APT.split()])
    if r.returncode != 0:
        print(ERR.format("install failed. Run this once in your terminal, then rerun:"))
        print(ERR.format("    sudo apt-get update && sudo apt-get install -y " + APT))
        sys.exit(1)


def install_runtime():
    ensure_venv()
    if os.path.exists(PY):
        ready = subprocess.run([PY, "-c", "import llama_cpp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        saved = backend_tag()
        if ready and saved in ("vulkan", "cpu"):
            print(f"runtime already installed ({saved}), skipping build")
            return saved
    for pkg in ("cmake", "huggingface_hub"):
        sh([os.path.join(BIN, "pip"), "install", "-q", pkg], check=True)
    env = dict(os.environ)
    env["PATH"] = f"{BIN}:{env['PATH']}"
    backend = "vulkan"
    if vulkan_sdk_ready():
        env["CMAKE_ARGS"] = "-DGGML_VULKAN=ON"
        print("building llama-cpp-python with Vulkan backend (3-6 min)...")
        r = sh([os.path.join(BIN, "pip"), "install", "--force-reinstall", "--no-cache-dir", "--no-deps",
                os.environ.get("DS_LCPP", "llama-cpp-python")], env=env)
        if r.returncode == 0:
            return backend
        print(WARN.format("Vulkan build failed, falling back to CPU build"))
    backend = "cpu"
    env.pop("CMAKE_ARGS", None)
    sh([os.path.join(BIN, "pip"), "install", "--force-reinstall", "--no-cache-dir", "--no-deps",
        os.environ.get("DS_LCPP", "llama-cpp-python")], env=env, check=True)
    return backend


def hf_token_from_zshrc():
    try:
        with open(os.path.expanduser("~/.zshrc")) as f:
            for line in f:
                if line.startswith("export HF_TOKEN="):
                    return line.split("=", 1)[1].strip().strip("'\"")
    except OSError:
        pass
    return ""


def model_path(quant):
    repo_id, fname, size = FILES[quant]
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if "HF_TOKEN" not in os.environ:
        token = hf_token_from_zshrc()
        if token:
            os.environ["HF_TOKEN"] = token
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sh([os.path.join(BIN, "pip"), "install", "-q", "huggingface_hub"], check=True)
        from huggingface_hub import hf_hub_download
    path = None
    for attempt in range(1, 4):
        try:
            path = hf_hub_download(repo_id=repo_id, filename=fname)
            break
        except Exception as e:
            print(WARN.format(f"download attempt {attempt}/3 failed: {e}"))
            time.sleep(3)
    if not path:
        raise SystemExit("download failed after 3 attempts; rerun install to resume")
    actual = os.path.getsize(path)
    if actual < size * 0.9:
        raise SystemExit(f"{fname} incomplete ({actual/1e9:.1f} GB); delete ~/.cache/huggingface and rerun install")
    return path


def mem_available():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) * 1024
    except OSError:
        pass
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def saved_quant():
    try:
        return open(QUANT_FILE).read().strip()
    except FileNotFoundError:
        return ""


def pick_quant(quant):
    s = quant or saved_quant()
    if s in FILES:
        return s
    avail = mem_available()
    return "Q4_K_M" if avail >= 15e9 else "Q3_K_M"


def backend_tag():
    try:
        return open(BACKEND_FILE).read().strip()
    except FileNotFoundError:
        return ""


def threads():
    return os.cpu_count() or 4


def cmd_status(args):
    total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    avail = shutil.disk_usage(BASE).free / 1e9
    print(f"cpu: {os.cpu_count()} cores | ram: {total:.0f} GiB | disk free: {avail:.0f} GB")
    print(f"vulkan sdk (build): {'ready' if vulkan_sdk_ready() else 'MISSING'}")
    print(f"venv: {'ok' if os.path.exists(PY) else 'absent'}")
    q = pick_quant(args.quant)
    print(f"selected quant: {q} ({FILES[q][2]/1e9:.1f} GB)  -- override with --quant")


def cmd_install(args):
    install_sdk()
    backend = install_runtime()
    with open(BACKEND_FILE, "w") as f:
        f.write(backend + "\n")
    print(f"downloading model (~{FILES[args.quant or pick_quant(args.quant)][2]/1e9:.1f} GB)...")
    p = model_path(pick_quant(args.quant))
    print(OK.format(f"model ready: {p}"))
    print(OK.format(f"backend: {backend}  | next: python3 manage_deepseek.py chat"))


def load_llm(p, args, verbose=False):
    from llama_cpp import Llama
    opts = dict(model_path=p, n_ctx=args.ctx, n_threads=threads(), verbose=verbose)
    try:
        llm = Llama(n_gpu_layers=-1, **opts)
        return llm, True
    except Exception:
        llm = Llama(n_gpu_layers=0, **opts)
        return llm, False


def cmd_chat(args):
    p = model_path(pick_quant(args.quant))
    tag = backend_tag()
    if tag:
        print(f"[backend: {tag}]")
    llm, gpu = load_llm(p, args)
    if not gpu:
        print(WARN.format("GPU load failed; running on CPU. Close other apps for speed."))
    system = "You are DeepSeek Coder, an AI coding assistant running locally."
    hist = []
    print("DeepSeek-Coder-V2-Lite-Instruct. Ctrl-D or /quit to exit.\n")
    while True:
        try:
            q = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("/quit", "/exit"):
            break
        hist.append(("User", q))
        prompt = system + "\n\n" + "\n\n".join(f"{r}: {t}" for r, t in hist) + "\n\nAssistant: "
        out = []
        for tok in llm(prompt, max_tokens=args.max_tokens, stream=True, stop=["User:", "Assistant:"]):
            s = tok["choices"][0]["text"]
            out.append(s)
            print(s, end="", flush=True)
        print("\n")
        hist.append(("Assistant", "".join(out).strip()))


def cmd_verify(args):
    p = model_path(pick_quant(args.quant))
    saved_fd = os.dup(2)
    tmp = tempfile.TemporaryFile()
    os.dup2(tmp.fileno(), 2)
    t0 = time.time()
    try:
        llm, _ = load_llm(p, args, verbose=True)
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        tmp.seek(0)
        idx = tmp.read().decode(errors="replace")
    load = time.time() - t0
    vk_lines = [l.strip() for l in idx.splitlines() if "Vulkan" in l or "ggml_vulkan" in l]
    print(OK.format(f"model loaded in {load:.1f}s"))
    for line in vk_lines:
        print(" " + line)
    t0 = time.time()
    out = llm("def fib(n):\n    return n if n < 2 else n - 1 + 2\nprint(fib(10))\n", max_tokens=48, stream=False)
    dt = time.time() - t0
    n = out["usage"]["completion_tokens"]
    label = "GPU" if vk_lines else "CPU"
    print(OK.format(f"completion: {n} tokens in {dt:.1f}s = {n/dt:.1f} tok/s ({label})"))


def _est_chars(msgs):
    return sum(len(str(m.get("content") or "")) for m in msgs)


SYSTEM_MAX = int(os.environ.get("SYSTEM_MAX", "1050"))
SHORT_SYSTEM = (
    "You are opencode, an interactive CLI tool that helps users with software engineering "
    "tasks. The declared tools let you read, edit, search, run commands, and delegate. "
    "Take direct action on the request now: use the tools, never just talk about it; "
    "verify your work by running or checking.")


def short_system(msg):
    """opencode's ~5.7k-char system prompt makes 7-8B models enter greeter-mode
    (they stop calling tools). Once it exceeds SYSTEM_MAX chars, replace it with
    the short action directive above; size is preserved, only behavior is tuned."""
    s = str(msg.get("content") or "")
    if len(s) <= SYSTEM_MAX:
        return msg
    return {"role": msg.get("role", "system"), "content": SHORT_SYSTEM}


def toolup_proxy(backend_base):
    """HTTP proxy that translates llama-server's raw tool-call output into the
    OpenAI tool_calls format opencode expects (fixes finish_reason=tool_calls)."""
    from agent import tool_calls_from_content

    class H(BaseHTTPRequestHandler):
        def _bw(self, path, body=None, headers=None):
            h = {k: v for k, v in (headers or {}).items()
                 if k.lower() in ("content-type", "authorization")}
            h.setdefault("Content-Type", "application/json")
            req = urllib.request.Request(backend_base + path, data=body, headers=h)
            return urllib.request.urlopen(req, timeout=600)

        def _passthrough(self, body=None):
            r = self._bw(self.path, body, dict(self.headers))
            self.send_response(r.status)
            self.send_header("Content-Type", r.headers.get("Content-Type", "text/event-stream"))
            self.end_headers()
            while True:
                chunk = r.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
            r.close()

        def do_GET(self):
            try:
                self._passthrough()
            except Exception as e:
                self.send_error(502, f"{e}")

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            try:
                payload = json.loads(body or b"{}")
            except Exception:
                payload = {}
            if self.path == "/v1/chat/completions" and isinstance(payload.get("messages"), list):
                payload["messages"] = [short_system(m) for m in payload["messages"]]
                tools_chars = len(json.dumps(payload.get("tools") or {}, ensure_ascii=False))
                self.log_message("req msgs=%d ch=%d tools_chars=%d last=%r",
                                 len(payload["messages"]), _est_chars(payload["messages"]), tools_chars,
                                 str((payload["messages"][-1].get("content") or "")[:60]))
                body = json.dumps(payload).encode()
                try:
                    with open(os.path.join(BASE, "lastreq.json"), "w") as _f:
                        _f.write(json.dumps(payload, indent=1))
                except OSError:
                    pass
            try:
                if self.path == "/v1/chat/completions" and payload.get("tools"):
                    bt = dict(payload)
                    bt["stream"] = False
                    r = self._bw("/v1/chat/completions", json.dumps(bt).encode(), dict(self.headers))
                    data = json.loads(r.read())

                    calls = [c for c in tool_calls_from_content(
                        data["choices"][0]["message"].get("content") or "") if c[1]]
                    if calls:
                        if payload.get("stream"):
                            chunk = {
                                "id": data["id"], "object": "chat.completion.chunk",
                                "created": data.get("created"), "model": data.get("model"),
                                "choices": [{"index": 0, "delta": {"role": "assistant", "content": None,
                                    "tool_calls": [{"index": i, "id": cid, "type": "function",
                                        "function": {"name": name, "arguments": args}}
                                       for i, (cid, name, args) in enumerate(calls)]},
                                    "finish_reason": "tool_calls"}],
                                "usage": data.get("usage"),
                            }
                            self.send_response(200)
                            self.send_header("Content-Type", "text/event-stream")
                            self.send_header("Cache-Control", "no-cache")
                            self.end_headers()
                            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                            self.wfile.write(b"data: [DONE]\n\n")
                            return
                        data["choices"][0]["message"]["tool_calls"] = [
                            {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}
                            for cid, name, args in calls]
                        data["choices"][0]["message"].pop("content", None)
                        data["choices"][0]["finish_reason"] = "tool_calls"
                        out = json.dumps(data).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(out)))
                        self.end_headers()
                        self.wfile.write(out)
                        return
                self._passthrough(body)
            except Exception as e:
                try:
                    self.send_error(502, f"{e}")
                except Exception:
                    pass

        def log_message(self, fmt, *args):
            print("[proxy] " + (fmt % args if args else fmt), file=sys.stderr, flush=True)

    return H


def cmd_serve(args):
    p = model_path(pick_quant(args.quant))
    print(f"serving {p}\n  toolup proxy: http://{args.host}:{args.port}/v1 (kompact in front on :8000)")
    native_dir = os.environ.get("LLAMA_SERVER_DIR") or os.path.join(BASE, "llama", "llama-native")
    native = os.path.join(native_dir, "llama-server")
    if os.path.exists(native):
        backend_port = 8002
        proxy = ThreadingHTTPServer((args.host, args.port), toolup_proxy(f"http://127.0.0.1:{backend_port}"))
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        tpl_file = os.environ.get("LLAMA_CHAT_TEMPLATE_FILE") or (
            os.path.join(BASE, "llama", "hermes2pro-tool.jinja") if "Hermes-2-Pro" in os.path.basename(p) else None)
        rope = ["--rope-scaling", "yarn", "--yarn-orig-ctx", "8192"] if ("Hermes-2-Pro" in os.path.basename(p) and args.ctx > 8192) else []
        tpl_arg = ["--chat-template", os.environ["LLAMA_CHAT_TEMPLATE"]] if os.environ.get("LLAMA_CHAT_TEMPLATE") else []
        f_arg = ["--jinja", "--chat-template-file", tpl_file] if tpl_file else []
        subprocess.run([
            native, "-m", p, "-c", str(args.ctx), "-ngl", "99", "--alias", os.path.basename(p),
            "--host", "127.0.0.1", "--port", str(backend_port), "--no-webui",
        ] + tpl_arg + f_arg + rope)
        return
    sh([os.path.join(BIN, "pip"), "install", "-q", "llama-cpp-python[server]"], check=True)
    subprocess.run([
        PY, "-m", "llama_cpp.server", "--model", p, "--n_ctx", str(args.ctx),
        "--n_threads", str(threads()), "--n_gpu_layers", "-1",
        "--host", args.host, "--port", str(args.port),
    ])


def main():
    ap = argparse.ArgumentParser(prog="manage_deepseek.py", description=__doc__)
    ap.add_argument("command", choices=["status", "install", "chat", "verify", "serve"])
    ap.add_argument("--quant", choices=list(FILES), help="default: Q4_K_M if RAM>=16GB else Q3_K_M")
    ap.add_argument("--ctx", type=int, default=4096, help="context window (tokens)")
    ap.add_argument("--max-tokens", type=int, default=512, help="chat stop length")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    if args.quant:
        with open(QUANT_FILE, "w") as f:
            f.write(args.quant)

    if args.command in ("install", "chat", "verify", "serve") and sys.executable != PY:
        if not os.path.exists(PY):
            ensure_venv()
        os.execv(PY, [PY, *sys.argv])

    {"status": cmd_status, "install": cmd_install, "chat": cmd_chat,
     "verify": cmd_verify, "serve": cmd_serve}[args.command](args)


if __name__ == "__main__":
    main()