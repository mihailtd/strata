# The serving path is not the bottleneck — measured, not assumed

§28 showed this rig is host-bound, which makes the streaming serving path an
obvious suspect: `server.py` + `cuda_graph.py` do a GPU sync, a Python BPE decode,
three nested pydantic constructions, a `model_dump()`, a `json.dumps()` and an
event-loop tick **per token**. It looks expensive. It is not.

## Per-token host cost (no GPU needed)

| operation | us |
| :--- | ---: |
| `tokenizer.decode([id])` | 2.70 |
| pydantic construct | 3.18 |
| construct + `model_dump()` | 6.55 |
| construct + dump + `json.dumps()` | **11.15** |
| `"<think>" in piece` | 0.03 |

**Serialisation tail ~= 13.8 us/token = 0.04% of a 35.51 ms token.**

## End to end

| arm | tok/s | vs |
| :--- | ---: | ---: |
| decoder, batch decode at end | 28.98 | 1.000x |
| decoder, `generate_tokens_stream` | 29.39 | 1.014x |
| decoder, stream + pydantic + json | 30.01 | 1.036x |
| **live server, non-streaming** | **26.90** | **0.928x** |
| **live server, streaming** | **25.76** | **0.889x** |

The streaming arms measuring *faster* than batch is ordering noise; the honest
reading is no measurable cost. TTFT is 46 ms on a cold expert (45.98 of it the
swap) and **0 ms** once resident.

## The distinction that matters

**"Host-bound" in §28 means GPU kernel-DISPATCH bound, not Python-application
bound.** A graph replay costs ~3.85 ms (1 layer) to ~34 ms (32 layers); tokenizer
and JSON work costs ~14 us. Three orders of magnitude apart. `.item()` per token
is likewise free — the CPU would be blocked on the GPU regardless.

Do not optimise the tokenizer, the pydantic models, or the SSE framing. The
server already delivers **93%** of what the decoder can produce, and the decoder
is at this hardware's graph-replay ceiling.
