"""Second pass: re-classify weak traces (Other/unclear, fallback, Unclassified) using the
actual request extracted from inside long first messages (e.g. "First Observation: ...
message='...'", "Task Description: ..."). Writes .cache/reclass.json, read by pipeline.py."""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from pipeline import assign_batch, norm_assign
from trace_utils import _as_list, message_text

PATTERNS = [r"First Observation:\s*sender='user'\s*message='(.{10,400}?)'",
            r"Task Description:\s*(.{10,400})", r"Task from supervisor:\s*(.{10,400})",
            r"New task:\s*Task:\s*(.{10,400})", r"<instructions>(.{10,400})", r"Task:\s*(.{10,400})"]


def key_request(msgs_json):
    texts = [message_text(m) for m in _as_list(msgs_json) if isinstance(m, dict)
             and m.get("role") in ("user", "human")]
    full = "\n".join(texts)
    for p in PATTERNS:
        m = re.search(p, full, re.S)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()[:350]
    # otherwise the second user turn is usually the customer's request
    return re.sub(r"\s+", " ", texts[1] if len(texts) > 1 else full[-350:]).strip()[:350]


def main(data="../data/traces.parquet", workers=12, batch=20):
    t = pd.read_pickle(".cache/features.pkl")
    a = pd.read_json("../output/trace_assignments.json")
    weak = a[(a.workflow.isin(["Other / unclear", "Unclassified"])) | (a.assigned_by != "glm")].trace_id.tolist()
    print("weak traces:", len(weak))
    df = pd.read_parquet(data, columns=["trace_id", "call_index", "input_messages_json"],
                         filters=[("trace_id", "in", weak)])
    df = df.sort_values("call_index").groupby("trace_id").last()
    tt = t.set_index("trace_id").loc[weak]
    texts = []
    for tid in weak:
        r = tt.loc[tid]
        texts.append(f"tools_available({len(r.tools)}): {', '.join(r.tools[:15])}\n"
                     f"tools_called: {', '.join(f'{k}x{v}' for k, v in list(r.called.items())[:8])}\n"
                     f"REQUEST: {key_request(df.loc[tid].input_messages_json)}")
    taxonomy = json.loads(Path(".cache/taxonomy.json").read_text(encoding="utf-8"))
    batches = [list(range(i, min(i + batch, len(weak)))) for i in range(0, len(weak), batch)]

    def work(b):
        try:
            return b, assign_batch(taxonomy, [texts[i] for i in b])
        except Exception as e:
            print("  batch error:", str(e)[:100], flush=True)
            return b, {}

    out = {}
    with ThreadPoolExecutor(workers) as ex:
        for n, (b, res) in enumerate(ex.map(work, batches), 1):
            for j, i in enumerate(b):
                uc, wf, c = norm_assign(taxonomy, res.get(j))
                if uc != "Unclassified":
                    out[weak[i]] = [uc, wf, c]
            print(f"  batch {n}/{len(batches)}", flush=True)
    Path(".cache/reclass.json").write_text(json.dumps(out), encoding="utf-8")
    print("reclassified:", len(out))


if __name__ == "__main__":
    main(*sys.argv[1:2])
