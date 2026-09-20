# Pitfalls

Each pitfall is expanded as symptom / root cause / fix / how we found it. The order is the order we hit them.

## 1. Startup error: `QSA sequence length 1000000 exceeds the configured limit 262144`

**Symptom.** The engine fails to start with a traceback in `mtp.py` complaining that the QSA sequence length 1000000 exceeds a configured limit of 262144, even though `--max-model-len 1000000` is set on the main model. ("QSA" is the abbreviation as it appears in the `mtp.py` traceback; the expansion — query-string-attention or otherwise — is not defined in the recorded source.)

**Root cause.** This vLLM build sizes the MTP draft model from `speculative_config.max_model_len`, whose default is `None`, which the draft then resolves to `262144`. The draft is constructed via `ModelConfig(max_model_len=speculative_config.max_model_len, hf_overrides=compose_draft_hf_overrides())`; a dict `hf_overrides` is not passed verbatim to the draft, so the main model's `--hf-overrides` and `--max-model-len` do not reach the draft's attention limit. The vLLM source version, full traceback, and before/after launch logs for this path are not recorded.

**Fix.** Set `"max_model_len": 1000000` **inside** `--speculative-config`, alongside `method`, `num_speculative_tokens`, etc. Do not rely on the top-level `--max-model-len` to propagate to the draft.

**How we found it.** The 1M stack failed to launch on the first bring-up with this exact traceback; the message names `mtp.py` and the two numbers (1000000 vs 262144), which pointed straight at the draft's attention limit rather than the main model. Reading the launch command showed `--max-model-len 1000000` was present but `--speculative-config` carried no `max_model_len` key.

## 2. Long repeated-output episodes only when MTP is on

**Symptom.** A completion degenerates into a repeated token string (the same word repeated until `max_tokens` 4096 is exhausted), taking about 200 s at roughly 20 tok/s. This happens only on configs where MTP is active (A and D); it does not happen with MTP off (C).

**Root cause.** The observed pattern is consistent with a vLLM async scheduling × MTP host/device race in which accepted-token counts bleed between concurrent requests (hypothesis from the A/B contrast, not isolated to source). The vLLM build in use was observed to run async scheduling whenever MTP was active (`async_scheduling=None` → observed auto-on with MTP), so any MTP use in this build inherited the behavior. This matches community reports #53912/#51571 (different stack: async on 16/288 broken, async off 0/288 broken; the upstream repository, fixed version, and accompanying excerpt are not recorded in the source).

**Fix.** Add one launch flag: `--no-async-scheduling`. Result: 0/120 runaways, per-run c7a wall clock back from 212–217 s to 48–52 s.

**How we found it.** The four-config single-variable isolation (120 questions per config) showed the runaway count tracks the async flag, not the draft-token count: A (MTP4, async on) = 3/120, B (MTP4, async off) = 0/120, C (MTP off) = 0/120, D (MTP3, async on) = 1/120. C removes MTP entirely, so it cannot separate the two; A vs B is the clean single-variable contrast and it points at async.

## 3. "But we never enabled async scheduling"

**Symptom.** The runaway loops appear even though `async_scheduling` was never set in the launch command. Operators assume async scheduling is off because they did not turn it on.

**Root cause.** `async_scheduling=None` (the unset state) was observed to behave as **auto-on** with MTP in this build — it is not the same as `async_scheduling=False`. The flag was observed to be opt-out, not opt-in, when MTP was used.

**Fix.** Always pass `--no-async-scheduling` explicitly when using MTP on this build. Do not assume the default is safe.

**How we found it.** The launch command for config A had no async-related flag, yet the runaway rate was 3/120; config B, identical except for the added `--no-async-scheduling`, dropped to 0/120. One consistent explanation is that async was already on by default; the parsed config or launch log that would confirm the default was not recorded, so the 3→0 change alone does not prove the default was on.

## 4. c7-agentic-if still 81.7 vs 95 after the fix

**Symptom.** After applying `--no-async-scheduling`, the runaway loops are gone (0/120) and all three wall-clock gates plus all four performance gates pass — but the c7-agentic-if score gate (≥90.0) still fails: candidate 81.7 vs baseline 95.0.

**Root cause.** The pattern is consistent with a model restraint behavior — suspected, not isolated to a model cause. The per-question breakdown shows the candidate over-calls tools: sub-category D Restraint & Refusal is 2/4 vs the baseline's 4/4, and every loss reason is "Extra tool calls made" (except C7A-28, a terminal-marker miss). Example: C7A-29 turn counts candidate [4, **16**, 4] vs baseline [3,3,3] — one run burned 16 tool-call turns. These runs had zero engine runaways; the loop bug and the restraint gap are different failure modes. Request/response transcripts, tool definitions, and the grader trace were not recorded, so engine/parse/scoring causes are not ruled out.

**Fix.** Do not chase this in the serving stack; it is out of scope for an engine-flag fix. The recorded mitigation is to keep the baseline model for c7-agentic-if, or to treat the restraint gap as unresolved pending an independent re-review.

**How we found it.** The per-question c7-agentic-if breakdown (config B post-fix, 3 runs) localized the loss to Restraint & Refusal and the "Extra tool calls made" loss reason, while the same runs showed zero engine runaways and wall clock with a median of 52 s inside the ≤1.5× (63 s) gate (one run at 67 s exceeded the per-run cap). A loop bug would show runaway text; this showed too many *correct* tool calls.

## 5. A clean synthetic reproducer → false confidence

**Symptom.** The synthetic multi-prefill reproducer (concurrency 4 × 10 batches) reports 0/40 runaways across all four configs, suggesting the bug is gone or never reproduced. Meanwhile the real agentic evaluation runs 3/120 on config A.

**Root cause.** The bug did not reproduce on this scripted batch prefill; it needed real concurrent agentic traffic (variable request mix, real tool-call turns) to trigger the accepted-token-count bleed. Other synthetic loads were not tested. The reproducer also did not exhibit community step-size bug #55375 in this sample (its PLE path `copy_`s the state index into a contiguous slot table), so it would have been clean even on that unrelated bug; source/patch evidence for the absence is not recorded.

**Fix.** Validate speculative-decoding bugs on real agentic workloads, not synthetic batches. Use `scripts/runaway_detector.py` on recorded chat completions (flags completion > 1500 tokens, or wall > 60 s, or a run of ≥ 20 identical whitespace-separated tokens) — it flags the cases recorded as runaway in the isolation set while the multi-prefill reproducer flags none in this sample. Raw logs and human labels for a full recall count were not recorded.

**How we found it.** The reproducer was run as part of every isolation config (the harness runs it after the eval-bank probes over the two freshest result dirs). It returned 0/40 on all four configs, including A where the real evaluation had 3/120 runaways. The disagreement between 0/40 and 3/120 is what told us the reproducer was the wrong tool for this bug, not evidence the bug was absent.
