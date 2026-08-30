#!/usr/bin/env python3
"""Minimal coding-agent loop over the local llama.cpp /v1 endpoint."""
import json
import re
import subprocess
import sys
import urllib.request

BASE = "http://127.0.0.1:8000/v1"

TOOLS = [
    {"type": "function", "function": {"name": "python",
      "description": "Run python code in a subprocess (30s timeout). Returns stdout/stderr. Use for anything executable: tests, math, file munging.",
      "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "read_file",
      "description": "Read a text file, first 8000 chars.",
      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file",
      "description": "Write text to a file (overwrites).",
      "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
]


def call_api(payload):
    req = urllib.request.Request(BASE + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=180))


def model_id():
    return json.load(urllib.request.urlopen(BASE + "/models"))["data"][0]["id"]


def run_tool(tc):
    name = tc["function"]["name"]
    args = json.loads(tc["function"]["arguments"])
    for k in ("code", "content"):
        if k in args:
            args[k] = args[k].replace("\\n", "\n")
    if name == "python":
        try:
            r = subprocess.run([sys.executable, "-c", args["code"]], capture_output=True, text=True, timeout=30)
            return (r.stdout + "\n" + r.stderr).strip()[:8000] or "(no output)"
        except subprocess.TimeoutExpired:
            return "(timed out after 30s)"
    if name == "read_file":
        try:
            return open(args["path"]).read()[:8000]
        except Exception as e:
            return f"error: {e}"
    if name == "write_file":
        open(args["path"], "w").write(args["content"])
        return "written"
    return "unknown tool"


def json_blocks(s):
    blocks, i, n = [], 0, len(s)
    while i < n:
        j = s.find("{", i)
        if j < 0:
            break
        depth, k, ins, esc = 0, j, False, False
        while k < n:
            c = s[k]
            if ins:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    ins = False
            elif c == '"':
                ins = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        blocks.append(s[j:k + 1])
        i = k + 1
    return blocks


def tool_calls_from_content(content):
    out = []
    def add(name, args):
        out.append((f"call_{len(out)}", name, args if isinstance(args, str) else json.dumps(args)))
    for name, args in re.findall(r"<function\s+name=\"([^\"]+)\"\s+arguments='([^']+)'\s*/?>", content or ""):
        add(name, args)
    for blk in json_blocks(content or ""):
        obj = None
        try:
            obj = json.loads(blk)
        except Exception:
            try:
                obj = json.loads(blk.replace("\n", "\\n"))
            except Exception:
                pass
        for o in (obj if isinstance(obj, list) else [obj]):
            if isinstance(o, dict):
                if "function" in o:
                    o = o["function"]
                if "name" in o and "arguments" in o:
                    add(o["name"], o["arguments"])
    return out


def find_tool_calls(m):
    payloads = []
    for c in m.get("tool_calls") or []:
        payloads.append((c["id"], c["function"]["name"], c["function"]["arguments"]))
    if payloads:
        return payloads, False
    return tool_calls_from_content(m.get("content") or ""), True


def main():
    msgs = [{"role": "user", "content": " ".join(sys.argv[1:]) or "hi"}]
    for _ in range(12):
        r = call_api({"model": model_id(), "messages": msgs, "tools": TOOLS})
        m = r["choices"][0]["message"]
        calls, from_content = find_tool_calls(m)
        if not calls:
            print(m.get("content") or "")
            return
        msgs.append({"role": "assistant", "content": m.get("content") or "",
                     "tool_calls": [{"id": cid, "type": "function",
                                     "function": {"name": name, "arguments": a}}
                                    for cid, name, a in calls]})
        for cid, name, a in calls:
            print(f"[tool] {name} {a}", flush=True)
            res = run_tool({"function": {"name": name, "arguments": a}})
            print(res[:400], flush=True)
            msgs.append({"role": "tool", "tool_call_id": cid, "content": res})


# ponytail: python tool is unsandboxed code exec on this host — fine for a personal toy agent,
# add a container/monitor sandbox when prompts come from anyone but you.
if __name__ == "__main__":
    main()