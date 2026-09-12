# Harness head-to-head session logs

Rescued from the old `scratch/harness_head_to_head_sandboxes/` tree (removed — it
was 64MB of disposable per-run sandbox scaffolding: generated FastAPI project
code, `.venv`s, lockfiles) when only these 128KB of actual session transcripts
were worth keeping.

Each file is one `session.jsonl.zstd` produced by
`benchmarks/harness_sdk/run_harness_sdk_head_to_head.py` for a `<model-config>__<task>` run
— the DSH harness agent's transcript for that head-to-head comparison. Decode
with `zstd -d`.

Re-running `run_harness_sdk_head_to_head.py` regenerates fresh sandboxes under
`scratch/harness_head_to_head_sandboxes/` again (that script's `TEMP_ROOT`
hasn't been repointed — pending the wider benchmarks refactor); pull any
transcript worth keeping out into this directory the same way afterward.
