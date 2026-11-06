import json
from pathlib import Path
import pyarrow.parquet as pq

def iter_calls(path=None):
    path = path or Path(__file__).with_name("traces.parquet")
    for batch in pq.ParquetFile(path).iter_batches(batch_size=100):
        for row in batch.to_pylist():
            for key in ("input_messages_json", "output_messages_json", "tool_definitions_json", "extra_json"):
                row[key.removesuffix("_json")] = json.loads(row.pop(key))
            yield row

if __name__ == "__main__":
    calls = iter_calls()
    first = next(calls)
    print("First call:", first["trace_id"], first["call_index"], first["usage_basis"])
    print("Rows:", 1 + sum(1 for _ in calls))
