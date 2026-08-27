# Serving Qwen3.8-Flash-Next (NVFP4) on DGX Spark — GB10 / sm_121

Field notes from getting `Qwen3.8-Flash-Next` serving correctly on a pair of
NVIDIA DGX Sparks (GB10, sm_121) with SGLang, TP=2.

**The short version:** the vendor recipe boots on GB10 and then silently produces
garbage past a certain context depth. The fix is one flag. The rest of this
document is how we found that, and the traps between here and there — because the
failure is silent, and every obvious way of checking for it will tell you the
server is fine.

> Hardware here is 2x DGX Spark (GB10, 128 GB unified each) linked over their
> high-speed fabric, TP=2, `nnodes=2`. Single-Spark users should still care about
> the attention-backend finding; the memory numbers will differ.

---

## 1. The failure

Above a certain context depth the model returns long unbroken runs of `!`:

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

with `finish_reason: length`. It is deterministic at `temperature 0`.

**`!` is token id 0.** That is the whole diagnosis. `argmax(logits) == 0` is what
you get from a degenerate or NaN logits vector, so this is **numerical corruption
in the forward pass**, not the model rambling, not a sampler problem, and not
"reasoning ran too long". Check it yourself:

```python
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
t.encode("!", add_special_tokens=False)   # -> [0]
```

### It gets worse: the server does not recover

Once it has happened, the engine stays broken. Subsequent requests fail the same
way **at any depth**, including a 55-token "Say hello." that comes back with an
empty `content` field. `/flush_cache` does **not** clear it. Only a restart does.

This is what makes the bug expensive: one bad session poisons the server for
everyone after it, and nothing is logged.

### A free live detector

With speculative decoding on, a collapsed server pins to:

```
accept len: 1.00, accept rate: 0.00
```

The MTP draft head proposes plausible tokens, the target argmaxes to 0, so nothing
ever matches — forever. Healthy is roughly `accept rate 0.4-0.8`. Watch it with:

```bash
docker logs <container> 2>&1 | grep -a "Decode batch" | tail
```

---

## 2. The cause

Qwen3.8-Flash-Next uses a sparse attention path (QSA, with a DeepSeek-style
indexer: `indexer_budget 2048`, `indexer_compress_ratio 4`). On GB10 **neither**
of SGLang's sparse-decode kernels works:

| sparse decode path | behaviour on sm_121 |
|---|---|
| trtllm-gen (gated to sm_100 by `is_sm100_supported()`) | boots if you force the gate open, then silently emits token 0 at depth |
| FA4-CuTe packed varlen (the non-sm100 fallback) | fails MLIR codegen at boot — a 2D layout sliced with a 3D coord |

Forcing the gate open is a popular workaround. **It trades a loud failure for a
silent one**, which is strictly worse. Kernel-level validation at short shapes
(e.g. `kv=[2051,2,256]`) passes cleanly and cannot see this — the sparse path is
not engaged at that length.

Related upstream: sgl-project/sglang#36558.

Note the gate patch is still **required to boot at all**, even with QSA bypassed by
`--attention-backend` — the decode resolver is reached during startup regardless.
So on GB10 today you need both: the patch to boot, and the attention backend to be
correct.

### What does not work

`--kv-cache-dtype fp8_e4m3` allocates as advertised (KV 839,168 -> 1,593,024
tokens, `torch.float8_e4m3fn`) and serves short prompts, then **crashes the server**
at around 20k context. The QSA `q.dtype == k.dtype` assert is no longer the
blocker, but something else in the path is. Avoid it here.

---

## 3. The fix

**Bypass QSA globally.** Not just decode:

```
--attention-backend flashinfer
```

The distinction matters. `--decode-attention-backend flashinfer` overrides decode
only and leaves **prefill on QSA**; that configuration passes short probes and then
degrades in real use. Use the global flag.

That flag alone is sufficient for correctness — verified by bisect, with every
other flag at stock. Nothing else is needed.

### Do not add `--mamba-full-memory-ratio`

The Qwen3.8-27B recipe carries `--mamba-full-memory-ratio 4.21`, and it is correct
there. On Flash-Next it **costs you a third of your capacity**.

It raises the mamba pool from 122 to 218 slots and pays for it out of KV
(1,160,832 -> 839,168 tokens). KV is the binding constraint here, not mamba slots,
so the trade is backwards. Measured at 12 concurrent sessions, 70k context each:

| | default mamba (122 slots, 1.16M KV) | `--mamba-full-memory-ratio 4.21` (218, 839k) |
|---|---|---|
| p50 | **6.23s** | 44.45s |
| p95 | **38.47s** | 213.49s |
| turns >30s | **8%** | **59%** |
| worst-session median | **14.10s** | 54.84s |

Same hardware, same model, one flag. Check the two allocation lines at boot and
size for whichever pool binds first — for this model that is KV.

### Working configuration

```bash
python3 -m sglang.launch_server \
  --model-path <ORG>/Qwen3.8-Flash-Next-NVFP4 \
  --trust-remote-code \
  --tp 2 --nnodes 2 --node-rank $RANK \
  --dist-init-addr <FABRIC_IP>:29520 \
  --quantization modelopt_fp4 \
  --fp4-gemm-backend flashinfer_cutlass \
  --moe-runner-backend flashinfer_cutlass \
  --attention-backend flashinfer \
  --page-size 64 \
  --mamba-scheduler-strategy extra_buffer \
  --mamba-track-interval 64 \
  --chunked-prefill-size 4096 \
  --max-running-requests 16 \
  --context-length 262144 \
  --mem-fraction-static 0.90 \
  --allow-auto-truncate \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --reasoning-parser auto \
  --tool-call-parser qwen3_coder \
  --enable-metrics --enable-cache-report \
  --host 0.0.0.0 --port 8888
```

Pin NCCL/GLOO to the fast interface (`NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`);
on the management path we measured 5 MB/s and the model will not load in any
reasonable time.

---

## 4. Other GB10 traps

**`--moe-runner-backend` must be set explicitly.** `auto` resolves to
`flashinfer_trtllm` here and raises `NotImplementedError: Unsupported
moe_runner_backend for NVFP4 MoE` before you reach anything else. Use
`flashinfer_cutlass`. Note it is a *separate* flag from `--fp4-gemm-backend`.

**`reasoning_effort` silently defaults to `xhigh`.** The chat template resolves
`reasoning_effort|default('xhigh')` whenever `enable_thinking` is not false, and
SGLang exposes no server flag for it. Valid values are `xhigh | medium | low`.

Measured on three coding tasks at `max_tokens 4096`:

| effort | behaviour |
|---|---|
| `xhigh` (default) | **bimodal** — on one task it burned the entire 4096-token budget on thinking and returned an **empty `content`**; on two others it thought *less* than `medium` and gave the tersest answers |
| `medium` | best behaved, completed every task |
| `low` | fine, slightly thinner |

`medium` injects **no instruction at all** — the template's `elif` chain has no
`medium` branch — so it is "the model unsteered", not a midpoint. **Pin it:**

```json
"chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "medium"}
```

An unknown key in `chat_template_kwargs` is ignored silently, so a typo means you
benchmark a reasoning model against a non-reasoning one without noticing.

**Tool calls use a custom XML form**, converted by `--tool-call-parser
qwen3_coder`. Without the parser, `tool_calls` is `null` and the raw XML lands in
`content` — no agent harness will ever see a tool call.

---

## 5. How to verify — this part is not optional

**A degraded server keeps returning numbers**, and nothing in the response says
otherwise — no error, no warning, no failed request. A configuration can pass a
full probe suite and still break under real load.

Run a canary **before and after** every measurement. A measurement is only valid
if both pass.

```python
# canary.py — exit 0 healthy, 1 degraded
CHECKS = [("2+2, digit only.",              lambda t: "4" in t),
          ("Say the single word: banana",   lambda t: "banana" in t.lower()),
          ("Complete: the capital of France is", lambda t: "paris" in t.lower())]
# fail if: content is empty, a run of 10+ '!' appears, or the answer is wrong
```

Two failure modes it catches that ordinary checks do not:

- `content: ""` with a populated `reasoning_content` — looks like an empty reply,
  not an error
- a server that was healthy when you started the run and degraded during it

**Probe shape matters more than probe depth.** A single-shot cold prefill and a
multi-turn short extend on a cached prefix exercise different paths and fail at
different points, so a depth threshold measured with one shape does not transfer
to the other. Neither is "the" threshold; both are properties of the probe. Test
the shape of your actual workload.

Note also that `--decode-attention-backend` overrides only the `full_attention`
layers. This model has 48 layers of which **36 are `linear_attention`**
(mamba/GDN), carrying recurrent state across turns — which is why the global
`--attention-backend` is the flag that matters.

---

## 6. Measured results

2x DGX Spark, TP=2, NVFP4, NEXTN speculation, thinking off.

**Single stream:** ~43 tok/s at 24k context, ~39 tok/s at 84k — decode stays
remarkably flat with depth.

**Real agent workload** (coding agent, tool calls, 100k+ context): ran to natural
completion, produced a complete tested feature, full repo suite green
(193 files / 2054 tests), **zero** collapsed decode batches.

**Concurrency** (70k context per session, 300s window):

| sessions | p50 | p95 | turns >30s | TTFT p95 | verdict |
|---|---|---|---|---|---|
| 4 | 3.28s | 14.51s | 2% | 1.80s | comfortable |
| 8 | 5.89s | 29.08s | 3% | 2.63s | good |
| 10 | 5.33s | 29.24s | 4% | 4.05s | good |
| 12 | 6.23s | 38.47s | 8% | 2.21s | usable |
| 16 | 148.62s | 256.34s | **70%** | 107.59s | **KV cliff** |

The cliff is arithmetic, not a defect: KV is sized at 1,160,832 tokens, and
1,160,832 / 70,000 ≈ 16. TTFT is the tell — p95 goes from 2.2s to 107s, and prefix
cache-hit drops to 0%, i.e. sessions lose their cached prefix and re-prefill from
scratch. It degrades into thrashing, gracefully, without corrupting: the canary
still passes and no output is malformed.

Divide your KV token count by your working context to predict the wall. To move it,
raise KV — not the mamba pool.

**Watch p50 with suspicion.** From 8 to 10 sessions p50 is flat (5.89 → 5.91)
while p90 nearly doubles (13.57 → 23.16) and turns over 30s go 3% → 8%. A p50-keyed
capacity band will tell you the 10th user is free. They are not.

---

## Licence / status

Field notes, no warranty. Numbers are from one pair of machines, mostly n=1 per
cell. Corrections and contradicting data welcome.
