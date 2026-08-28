# Serving Qwen3.8-Flash-Next (NVFP4) on DGX Spark — GB10 / sm_121

Field notes from getting `Qwen3.8-Flash-Next` serving correctly on a pair of
NVIDIA DGX Sparks (GB10, sm_121) with SGLang, TP=2.

**The short version:** the vendor recipe boots on GB10 and then silently produces
garbage past a certain context depth. The fix is a working sparse-decode kernel;
no combination of stock flags is sufficient. The rest of this document is how we
found that, and the traps between here and there — because the failure is silent,
and every obvious way of checking for it will tell you the server is fine.

> ### Correction — 2026-08-28
>
> **An earlier version of this document said `--attention-backend flashinfer`
> alone was sufficient for correctness. That was wrong.** The flag is necessary
> but it does not remove QSA from the path: the boot log still reads
> `Using QSA for sparse full-attention layers`, because the flag only swaps the
> dense backend that QSA *wraps*. With the trtllm-gen gate patch still in place —
> which that version also told you to apply, in order to boot at all — the
> silently-wrong sparse decode remains live, and the `!!!!` collapse returns at
> depth.
>
> We missed it because the failure is **stochastic and depth-dependent**, and
> every rung of our matrix was n=1. Re-measured with repeats:
>
> | context depth | collapse rate (flag-only "fix") |
> |---|---|
> | 120k | 1/4 |
> | 160k | 1/4 |
> | 190k | 2/4 |
> | 210k | **4/4** |
>
> 120k is well inside ordinary agent use, so capping context is not a mitigation.
> See §3 for what actually fixes it.

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

We previously wrote that the gate patch is "required to boot at all". That is true
only if the *other* path is left broken. Repair the FA4-CuTe fallback instead (§3)
and the server boots with the gate untouched — which is the cleanest evidence that
forcing it open was never necessary, only convenient.

### What does not work

`--kv-cache-dtype fp8_e4m3` allocates as advertised (KV 839,168 -> 1,593,024
tokens, `torch.float8_e4m3fn`) and serves short prompts, then **crashes the server**
at around 20k context. The QSA `q.dtype == k.dtype` assert is no longer the
blocker, but something else in the path is. Avoid it here.

---

## 3. The fix

**Replace the sparse-decode kernel. Do not force the trtllm-gen gate open.**

The two stock paths both fail (§2), so the real fix is to repair the *intended*
fallback rather than un-gate the one NVIDIA disabled. MiaAI-Lab did exactly this:
they leave `is_sm100_supported()` gated off and give
`_resolve_flash_attn_varlen_func` a Triton varlen kernel that works on GB10.

> **Credit:** the kernel fix is theirs, not ours —
> <https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks>

Measured here at 210k context, NEXTN on, every other flag identical — the image is
the only variable:

| sparse decode | result at 210k | accept rate | decode |
|---|---|---|---|
| forced trtllm-gen + `--attention-backend flashinfer` | **4/4 collapse** | 0.72 | 43.2 tok/s |
| MiaAI-Lab Triton fallback | **6/6 clean** | 0.90 | **57.5 tok/s** |

It is correct *and* faster: a correct kernel drafts correctly, so the speculative
accept rate rises from 0.72 to 0.90. Their image also boots **without** the gate
patch, which stock cannot — direct confirmation that the fallback path is genuinely
repaired rather than bypassed.

### Still keep `--attention-backend flashinfer`

It remains necessary — `--decode-attention-backend flashinfer` overrides decode
only and leaves **prefill on QSA**; that configuration passes short probes and then
degrades in real use. Use the global flag. It is just no longer *sufficient*.

### What we tried that did NOT fix it

Recorded so nobody repeats them. All still collapse at 210k:

| attempt | result |
|---|---|
| `TL_DISABLE_WARP_SPECIALIZED` on the racy QSA decode kernel | 2/6 collapse |
| disabling QSA MTP IndexShare (`index_share_for_mtp_iteration`) | 3/6 collapse |
| `--chunked-prefill-size 1024` | still collapses |
| `--speculative-algorithm none` | clean, but costs 2.5x decode (43 -> 17.5 tok/s) |

Note the TileLang boot-time warning `Data race detected: Logits(bx, position)` is a
red herring — it is present on healthy and collapsing builds alike.

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

Run it on an image carrying the Triton QSA fallback (§3) — build it from the
MiaAI-Lab recipe. On the stock image this same command still collapses at depth.
Do **not** also bind-mount a trtllm-gate patch: it overwrites the very file the
fix patches and puts the broken kernel back.

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
  --chunked-prefill-size 1024 \
  --max-prefill-tokens 2048 \
  --max-running-requests 16 \
  --context-length 262144 \
  --mem-fraction-static 0.82 \
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

### Why chunk 1024 and fraction 0.82, not 4096 / 0.90

The QSA indexer allocates an `fp32 [chunk_size x history_length]` logits matrix
per sparse layer per chunk, plus gather/topk copies. At chunk 4096 and ~240k
history that transient is **~8-10 GB**. With `--mem-fraction-static 0.90` we had
8.9 GB free and saw **2/6 collapse at 240k** even on the fixed kernel; at 0.82
(17.9 GB free) with chunk 1024 the same test is **6/6 clean**.

This diagnosis is MiaAI-Lab's — see "the QSA indexer prefill workspace scales with
`chunk x history`" in their README. Do not raise chunk without profiling the peak
transient against your free budget.

It is not free. KV drops **1,209,792 -> 708,288 tokens** (~17 -> ~10 concurrent
sessions at 70k each), and deep prefill roughly doubles (~100s -> ~165s at 240k).
We took that trade deliberately: a correct answer slowly beats `!!!!` quickly.

---

## 4. Other GB10 traps

### There is a SECOND `!!!!` bug, with a different trigger

Do not assume a `!` run means the sparse-decode problem above. There are two:

| | this document's bug | [sglang#36537](https://github.com/sgl-project/sglang/issues/36537) |
|---|---|---|
| trigger | context depth (>=120k), stochastic | thinking ON **+** `tools` in the request **+** `--tool-call-parser qwen3_coder` |
| reproduces with thinking off | **yes** | no |
| fix | working sparse-decode kernel (§3) | none upstream; disable thinking for those sessions |

Both drop spec accept rate to 0.00 and both emit token 0, so the symptom is
identical. We had a field incident that matched *both* sets of conditions at once
and initially attributed it entirely to depth. Credit to
[MiaAI-Lab](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) for
documenting #36537 — their README is the reason we separated the two.

Workaround for #36537, per request (keeps thinking elsewhere):

```json
"chat_template_kwargs": {"enable_thinking": false}
```

**Check what your client actually sends.** Ours passed
`chat_template_kwargs {"thinking": false}` — the template reads `enable_thinking`,
unknown keys are silently ignored, so thinking stayed **on** and every tool-using
agent session sat on the #36537 trigger without anyone intending it. A key
typo here fails silently in the direction of the bug.

### Zero-token replies at the context ceiling

With `--allow-auto-truncate`, a prompt that fills the window returns
`finish_reason: "length"` with **`completion_tokens: 0`** and empty content — no
error, no warning. Clients render it as a blank reply. If you do not need
truncation, leave the flag off so oversized prompts fail loudly instead.


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

> **But know what it cannot see.** These prompts are short, and the sparse-decode
> collapse is depth-dependent. We have watched this canary pass three for three
> while the same server collapsed on **100%** of 210k-context requests. A green
> canary means "not globally wedged"; it says nothing about depth. Probe at the
> depth your workload actually reaches, and **repeat it** — see below.

**One sample per configuration is not a measurement.** The collapse is a
stochastic failure: at 190k it fires on roughly half of attempts, so a single
clean run is a coin flip you will read as a pass. This is precisely how the
earlier version of this document shipped a broken recommendation — every rung of
the matrix was n=1. Use n>=6 per cell and report the rate, not a verdict.

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
