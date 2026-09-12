# The captured graph owns the cache length counter

## What went wrong

A long request pinned the GPU at 100% for **4h20m** with no progress, ignored
`SIGINT`, and had to be `SIGKILL`ed. Short requests were fine. The wedge was
deterministic: same step, same position, every run.

## Root cause

`transformers`' `StaticLayer` keeps its KV write offset in a **device tensor** and
advances it in place:

```python
cache_position = torch.arange(kv_length, device=self.device) + self.cumulative_length
self.cumulative_length.add_(kv_length)
```

Both lines are **captured into the CUDA graph**. Every replay therefore advances
the counter by `width`, no matter what `cache_position` the caller copies in — the
attention layers derive their own offset and never read ours.

Autoregressive decode never notices: one replay, one token, counter and sequence
advance together, and `cache.reset()` per request keeps it under `max_cache_len`.

**Speculative decode replays overlapping ranges** — a K+1 verify, then a commit
re-forward at the *same* position. The counter does not rewind. So:

* **correctness** — committed tokens were written at drifting KV offsets, starting
  at the very first partial accept
* **stability** — the counter grows unbounded, runs past `max_cache_len`, the
  attention kernel indexes out of range, and the GPU wedges

## The evidence

`probe_replay_count_wedge.py` hammers replays at a **fixed** position with no
decoding at all, so nothing but replay count varies. The wedge point is exactly
`max_cache_len / width`:

| mode | tokens/replay | predicted | observed |
| :--- | ---: | ---: | ---: |
| `onlyw5` width K+1 | 5 | 410 | ~350–400 |
| `onlyw1` width 1 | 1 | 2048 | ~1950–2000 |
| real decode, 58-tok prompt | ~7.7/step | step ~266 | step **300–303** |
| real decode, 458-tok prompt | ~7.7/step | step ~206 | step **244** |

Ruled out first, each by measurement: sequence position (all 20 positions
including 891/1193/2000 replay fine from a fresh cache), the shared graph memory
pool (private pools per width changed nothing), the SSM snapshot/restore
(`onlyw5` carries neither), and decode itself (wedges with no draft head and no
token generation).

## The fix

`bucketed_speculative.py` pins the counter to the true position before a replay,
and only when it has drifted — after a verify the counter already sits where the
next verify wants it, so only a rollback pays. Costs **4%**.

After: 5,200 replays clean where it used to die at 350, and a 1,400-token decode
completes.

## The uncomfortable result

`benchmark_pinned_vs_graph.py`, 512 tokens, chat-template prompt:

| arm | tok/s | vs graph | tau |
| :--- | ---: | ---: | ---: |
| graph autoregressive (server today) | 25.51 | 1.000x | — |
| bucketed speculative, pinned (correct) | 19.91 | **0.781x** | 1.65 |
| bucketed speculative, unpinned (wrong) | 20.78 | 0.815x | 1.68 |

Speculation is **22% slower** than the path the server already runs. tau is 2.5
over the first ~60 tokens of predictable thinking preamble and falls to **1.65**
over a real-length answer, where a 5-token verify costs more than it saves.

The earlier +8.5% / +20.6% headlines were measured at **64 tokens** — inside the
inflated-tau window — *and* on corrupted KV offsets. Both are withdrawn.

## Run it

```bash
MODE=onlyw5 uv run --env-file .env python \
  benchmarks/runtime/speculative/cache_length_counter/probe_replay_count_wedge.py
N=512 uv run --env-file .env python \
  benchmarks/runtime/speculative/cache_length_counter/benchmark_pinned_vs_graph.py
```

`SPECULATIVE_PIN_CACHE_LEN=0` restores the old, incorrect path for attribution
only. It wedges the GPU past `2048/width` replays; never serve traffic with it.
