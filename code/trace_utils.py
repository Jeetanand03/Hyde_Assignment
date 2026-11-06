"""Helpers for traces.parquet: load, parse message JSON, extract text/tool calls."""
import json

import pandas as pd


def load_df(path="data/traces.parquet"):
    return pd.read_parquet(path)


def parse_json(s):
    if s is None or (isinstance(s, float) and pd.isna(s)) or s == "":
        return None
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return s


def _as_list(msgs):
    msgs = parse_json(msgs)
    if msgs is None:
        return []
    if isinstance(msgs, dict):
        msgs = msgs.get("messages", [msgs])
    return msgs if isinstance(msgs, list) else [msgs]


def _walk_text(x, out):
    """Collect text from str / content arrays / parts arrays of any shape."""
    if x is None:
        return
    if isinstance(x, str):
        out.append(x)
    elif isinstance(x, list):
        for y in x:
            _walk_text(y, out)
    elif isinstance(x, dict):
        t = x.get("type")
        if "text" in x and isinstance(x["text"], str):
            out.append(x["text"])
        elif t in ("tool_result", "tool_call_response") and "content" in x:
            _walk_text(x["content"], out)
        for k in ("content", "parts"):
            if k in x and t not in ("tool_result", "tool_call_response"):
                _walk_text(x[k], out)


def message_text(msg):
    out = []
    _walk_text(msg.get("content") if isinstance(msg, dict) else msg, out)
    if isinstance(msg, dict) and "parts" in msg:
        _walk_text(msg["parts"], out)
    return "\n".join(s for s in out if s)


def messages_text(msgs_json, max_chars=None):
    """Flatten a messages JSON column into 'role: text' lines."""
    lines = []
    for m in _as_list(msgs_json):
        role = m.get("role", "?") if isinstance(m, dict) else "?"
        txt = message_text(m)
        if txt:
            lines.append(f"{role}: {txt}")
    s = "\n".join(lines)
    return s[:max_chars] if max_chars else s


def tool_calls(msgs_json):
    """Return list of (tool_name, arguments) found anywhere in the messages."""
    calls = []

    def walk(x):
        if isinstance(x, list):
            for y in x:
                walk(y)
        elif isinstance(x, dict):
            t = x.get("type")
            if t in ("tool_use", "tool_call", "function_call") or "function" in x or "functionCall" in x:
                fn = x.get("function") or x.get("functionCall") or x
                name = fn.get("name") if isinstance(fn, dict) else None
                args = fn.get("arguments", fn.get("args", fn.get("input"))) if isinstance(fn, dict) else None
                if name:
                    calls.append((name, args))
            for k in ("tool_calls", "content", "parts"):
                if k in x:
                    walk(x[k])

    walk(_as_list(msgs_json))
    return calls
