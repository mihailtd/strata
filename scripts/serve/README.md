# scripts/serve/ — run it

| script | what it does |
| :--- | :--- |
| `run_openai_api_server.py` | OpenAI-compatible server, in-place expert folding |
| `test_openai_api_server.py` | server smoke test |
| `chat.py` | interactive CLI |

```bash
uv run --env-file .env python scripts/serve/run_openai_api_server.py
```

## Gotchas that have cost real time

- **Empty `content` is usually not a bug.** Thinking-mode models put the answer in
  `reasoning_content`; a client reading only `content` sees nothing and it looks
  like a broken stop-token. Check both fields before debugging the server.
- **`--env-file .env` is required** or the process runs CPU-only while looking fine.
- **Adapter folding is ~1.1–1.9 ms per swap**, in place, no VRAM bloat. If a swap
  appears to cost more than that, something is re-capturing graphs, not folding.
