# Trace Mining — Variant 06

**Hierarchy:** Business use case → Workflow → Trace (default). Session = `trace_id`; all calls of a trace inherit its assignment.

## How it works (`code/`)
1. **features.py** (deterministic): groups the 50,000 calls into 3,154 traces by `trace_id`/`call_index`. Each trace gets its available tools, the tools it actually called, and its opening requests. Long policy prompts are trimmed to head and tail.
2. **pipeline.py: taxonomy.** The GLM model (`hyde/glm-5.3-flash`) sees a sample stratified by tool set (up to 3 traces per distinct tool set). It proposes use cases and workflows (`output/taxonomy.json`).
3. **Assignment.** GLM classifies unique trace digests in batches of 25 into menu entries with confidence. Misses get their tool-set's majority label; tool evidence corrects vague/mislabelled traces (`assigned_by`: glm/fallback/tool_rule).
4. **Aggregation.** Traces, calls and tokens per use case and workflow, kept in three separate buckets:
   - **recorded**: `usage_basis=source_reported`. Shares are computed on this basis. Cache read/creation is shown as an informational subset of input.
   - **estimated_cl100k_base**: reported separately, never mixed into recorded totals.
   - **unknown**: missing counts, counted as calls rather than filled with zero.

   `source_reported_session_usage` is a session total. It is listed per trace and never added to call sums.

## Outputs (`output/`)
- `results.json`: hierarchy, metrics, usage shares, and 3 representative traces per workflow with short excerpts.
- `trace_assignments.json`: every `trace_id` (with its first `span_id`), its use case, workflow, confidence and token metrics.
- `index.html`: self-contained viewer with expandable use cases and workflows, share bars and examples. Open it in a browser.

## Run
Place the provided `traces.parquet` in `data/` (not committed, 82 MB).
```
pip install -r code/requirements.txt
# .env (not included): GLM_API_KEY, GLM_BASE_URL=https://inference.hyde.tech/v1, GLM_MODEL=hyde/glm-5.3-flash
cd code && python pipeline.py --data ../data/traces.parquet --out ../output
```

**reclassify.py** re-labels weak traces from the request extracted inside long prompts (`glm_pass2`).
