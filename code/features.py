"""Stage 1: deterministic per-trace features + token accounting (no LLM)."""
import hashlib
import json
import re
from collections import Counter

import pandas as pd

from trace_utils import _as_list, message_text, tool_calls

REDACT = re.compile(r"\s+")


def tool_names(defs_json):
    try:
        defs = json.loads(defs_json) or []
    except (TypeError, ValueError):
        return []
    out = []
    for t in defs if isinstance(defs, list) else []:
        if isinstance(t, dict):
            out.append(t.get("name") or (t.get("function") or {}).get("name") or "?")
    return sorted(set(out))


def user_requests(msgs_json, k=3):
    """First k user/system texts; long policy blocks are shortened to their head."""
    out = []
    for m in _as_list(msgs_json):
        if not isinstance(m, dict) or m.get("role") not in ("user", "system", "human"):
            continue
        t = REDACT.sub(" ", message_text(m)).strip()
        if not t:
            continue
        out.append(f"[{m.get('role')}] " + (t[:220] + " …" + t[-280:] if len(t) > 700 else t))
        if len(out) >= k:
            break
    return out


def session_usage(extra):
    try:
        e = json.loads(extra)
    except (TypeError, ValueError):
        return None
    return e.get("source_reported_session_usage")


def cache_parts(extra):
    try:
        e = json.loads(extra)
    except (TypeError, ValueError):
        return 0, 0
    return e.get("cache_read_input_tokens") or 0, e.get("cache_creation_input_tokens") or 0


def build(df):
    df = df.sort_values(["trace_id", "call_index"])
    # token columns split by usage_basis; NaN stays unknown (never zero-filled)
    for basis, tag in (("source_reported", "rec"), ("estimated_cl100k_base", "est")):
        m = df.usage_basis == basis
        df[f"{tag}_in"] = df.input_tokens.where(m)
        df[f"{tag}_out"] = df.output_tokens.where(m)
    df["unknown_call"] = (df.usage_basis == "unknown") | df.input_tokens.isna() | df.output_tokens.isna()
    cp = df.extra_json.map(cache_parts)
    df["cache_read"] = cp.map(lambda x: x[0])
    df["cache_create"] = cp.map(lambda x: x[1])

    rows = []
    for tid, g in df.groupby("trace_id", sort=False):
        first, last = g.iloc[0], g.iloc[-1]
        tools = tool_names(first.tool_definitions_json)
        called = Counter()
        for s in g.output_messages_json:
            for name, _ in tool_calls(s):
                called[name] += 1
        reqs = user_requests(last.input_messages_json) or user_requests(first.input_messages_json)
        su = next((x for x in g.extra_json.map(session_usage) if x), None)
        rows.append({
            "trace_id": tid,
            "n_calls": len(g),
            "session_call_count": int(first.session_call_count),
            "models": sorted(g.model.unique().tolist()),
            "usage_basis": sorted(g.usage_basis.unique().tolist()),
            "rec_in": g.rec_in.sum(min_count=1), "rec_out": g.rec_out.sum(min_count=1),
            "est_in": g.est_in.sum(min_count=1), "est_out": g.est_out.sum(min_count=1),
            "unknown_calls": int(g.unknown_call.sum()),
            "cache_read": int(g.cache_read.sum()), "cache_create": int(g.cache_create.sum()),
            "session_usage": su,
            "start": first.start_time, "end": last.end_time,
            "toolsig": hashlib.md5(str(tools).encode()).hexdigest()[:8],
            "tools": tools,
            "called": dict(called.most_common(12)),
            "requests": reqs,
            "first_span": first.span_id,
        })
    return pd.DataFrame(rows)


def digest(r, n=900):
    tools = r.tools[:25]
    s = (f"tools_available({len(r.tools)}): {', '.join(tools)}\n"
         f"tools_called: {', '.join(f'{k}x{v}' for k, v in r.called.items())}\n"
         + "\n".join(r.requests))
    return s[:n]
