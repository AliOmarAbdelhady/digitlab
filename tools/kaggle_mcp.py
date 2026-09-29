#!/usr/bin/env python3
"""Minimal client for the official Kaggle MCP server (https://www.kaggle.com/mcp).

Speaks Streamable-HTTP JSON-RPC directly, so the same calls the ZCode MCP
integration will make natively (after session restart) can be scripted here.

Usage:
  mcp_call.py tools/list
  mcp_call.py call <tool_name> '<json-args>'
  mcp_call.py schema <tool_name>     # pretty-print a tool's inputSchema
"""
import json
import pathlib
import re
import sys
import urllib.request

URL = "https://www.kaggle.com/mcp"


def token() -> str:
    cfg = json.loads((pathlib.Path.home() / ".zcode/cli/config.json").read_text())
    auth = cfg["mcp"]["servers"]["kaggle"]["headers"]["Authorization"]
    return auth.removeprefix("Bearer ")


def rpc(method: str, params: dict | None = None, msg_id: int = 1):
    payload = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        payload["params"] = params
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {token()}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as res:
        body = res.read().decode()
    # streamable http: reply may be SSE-framed ('event: message\ndata: {...}')
    m = re.search(r"^data: (.*)$", body, re.M)
    return json.loads(m.group(1) if m else body)


def parse_sse_multiline(body: str):
    """Some replies contain several data: lines — concat them."""
    chunks = re.findall(r"^data: (.*)$", body, re.M)
    return [json.loads(c) for c in chunks]


def call_tool(name: str, arguments: dict, msg_id: int = 2):
    return rpc("tools/call", {"name": name, "arguments": arguments}, msg_id)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    if cmd == "tools/list":
        result = rpc("tools/list")["result"]["tools"]
        for t in result:
            print(f"- {t['name']}: {t['description'].splitlines()[0]}")
        return 0
    if cmd == "schema":
        name = sys.argv[2]
        tools = rpc("tools/list")["result"]["tools"]
        t = next((x for x in tools if x["name"] == name), None)
        if not t:
            print(f"unknown tool {name}")
            return 1
        print(json.dumps(t.get("inputSchema", {}), indent=1))
        return 0
    if cmd == "call":
        name = sys.argv[2]
        args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        out = call_tool(name, args)
        text = json.dumps(out, indent=1, default=str)
        print(text[:20000])
        if out.get("isError"):
            return 1
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
