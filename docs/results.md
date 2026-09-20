# Results

All numbers verbatim from the measurement session. Every table carries a one-line note of the measurement condition above it. The evaluation bank is our private 11-category eval bank (questions not published); only category codes and item IDs appear here.

## Table 1 — Single-variable isolation

Condition: real scenarios c7a×3 + c3×2 = 120 questions per config; thinking off, greedy, parallel 4; harness v3.

| Config | Delta vs A | Runaway loops | c7a wall clock per run | c7a median score |
|---|---|---|---|---|
| A | MTP4, async auto-on (as-shipped default) | **3/120 (= 1/40)** | 212–217 s | 81.7 |
| B | **+ `--no-async-scheduling` (the fix)** | **0/120** | 48–52 s | 91.7 |
| C | MTP off | 0/120 | 58 s | not recorded |
| D | MTP3, async auto-on | **1/120** (that runaway: 8220 tokens / 330 s; the counting scope — single completion vs multi-turn aggregate — is not recorded in the source) | not recorded | not recorded |

Supplementary findings from the same isolation session:

- Synthetic multi-prefill reproducer (4-way × 10 batches): **0/40 on all four configs** in this sample — this build did not exhibit community step-size bug #55375 on the reproducer (its PLE path `copy_`s the state index into a contiguous slot table); source/patch evidence for the absence is not recorded.
- Community analogs #53912/#51571 (different stack): async on 16/288 broken, async off 0/288 broken; the upstream repository, fixed version, and accompanying excerpt are not recorded in the source.
- Baseline control: the DeepSeek V4 Flash Vision-Exp baseline (own speculation + async auto-on): **0 runaways in 330 questions** in this sample — the bug was not observed in this sample; it is not established that the baseline model is immune, and its speculation mechanism differs.
- Note on disagreement between sources: config B's c7a median is 91.7 in the isolation run (Table 1) but 81.7 (runs 85.0/80.0/81.7) in the later full-table re-adjudication (Table 2) — same config, different runs; both figures are reported.

## Table 2 — v3 full-table re-adjudication of config B

Condition: candidate = B stack, baseline = DeepSeek V4 Flash Vision-Exp; "perf" rows as recorded; perf units — decode/prefill/six-stream in tok/s, kv in tokens — match the K1 probe in Table 4; input/output length, cold/warm state, timing window, and aggregation rule are not recorded in the source.

| Category / gate | Baseline | Candidate (runs) | Gate | OK |
|---|---|---|---|---|
| c1-kbqa | 83.3 | 93.3 [90.0, 96.67] | ≥73.3 | ✓ |
| c2-longctx (crit) | 87.5 | 100.0 [100.0, 100.0] | ≥82.5 | ✓ |
| c3-tool (crit) | 66.7 | 85.0 [83.33, 86.67] | ≥61.7 | ✓ |
| c4-code | 87.5 | 95.8 [95.83, 95.83] | ≥77.5 | ✓ |
| c5-extract | 92.2 | 91.1 [92.28, 89.83] | ≥82.2 | ✓ |
| c6-vision (crit) | 82.5 | 85.0 [85.0, 85.0] | ≥77.5 | ✓ |
| c7-zhif | 73.3 | 85.0 [86.67, 83.33] | ≥63.3 | ✓ |
| **c7-agentic-if (crit)** | **95.0** | **81.7 [85.0, 80.0, 81.67]** | **≥90.0** | **✗** |
| c8-judgment (crit) | 78.3 | 75.0 [73.33, 76.67] | ≥73.3 | ✓ |
| c9-long-coding (crit) | 100.0 | 98.3 [100.0, 96.53] | ≥95.0 | ✓ |
| c10-sre-ops | 73.3 | 88.3 [86.67, 90.0] | ≥63.3 | ✓ |
| wall c3-tool | 39 s | 46 s (pre-fix async era: 122 s) | ≤1.5× (58.5 s; displayed as 59 s) | ✓ |
| wall c6-vision | 18 s | 17 s (pre-fix async era: 179 s) | ≤1.5× (27 s) | ✓ |
| wall c7-agentic-if | 42 s | 52 s (pre-fix async era: 214 s) | ≤1.5× (63 s) | ✓ |
| perf decode | 32.8 | 51.2 | ≥33 | ✓ |
| perf prefill | 2129 | 3159 | ≥2000 | ✓ |
| perf six (streams) | 82.8 | 87.6 | ≥75 | ✓ |
| perf kv | 1492180 | 3455574 | ≥1.5e+06 | ✓ |
| errors | 0 | 0 | — | — |

Δ own over shared categories = 91.7 − 85.0 = **+6.7** (the own/shared category sets, weighting, and the two totals' computation are not recorded in the source; the +6.7 figure is reported as recorded, not recomputable from this document). Verdict recorded in the source: negative result (baseline unchanged); the fix itself is validated by the three wall-clock gates moving from fail-adjacent to pass. Both nodes' kernel logs post-run: zero GPU Xid; `NVRM NV_ERR_NO_MEMORY 0x51` 5 lines per node with timestamps coinciding with engine startup (the source labels this a known startup memory-probe artifact; dmesg text and version-specific probe-path evidence were not recorded).

## Table 3 — Per-question c7-agentic-if breakdown

Condition: config B post-fix, non-thinking greedy, 3 runs; max 60 = 30 questions × 2.

| Sub-category | Candidate r1/r2/r3 | Baseline r1/r2/r3 |
|---|---|---|
| A Tool Selection | 20/20, 20/20, 18/20 | 20/20, 20/20, 20/20 |
| B Parameter Precision | 3/6, 3/6, 3/6 | 5/6, 5/6, 5/6 |
| C Multi-Step Chains | 6/6, 6/6, 6/6 | 6/6, 6/6, 6/6 |
| D Restraint & Refusal | 2/4, 2/4, 2/4 | 4/4, 4/4, 4/4 |
| E Error Recovery | 20/24, 17/24, 20/24 | 22/24, 22/24, 22/24 |
| **Total** | **51 / 48 / 49** | **57 / 57 / 57** (item-identical across runs) |

Losing items (candidate points, three runs): C7A-09 [2,2,0], C7A-17 [2,0,2], C7A-22 [0,0,0], C7A-26 [0,0,0], C7A-28 [2,1,2] (partial: answer never emits the expected terminal marker), C7A-29 [0,0,0]. Every loss reason is "Extra tool calls made" (except the C7A-28 marker miss). Runaway-style example: C7A-29 turn counts candidate [4, **16**, 4] vs baseline [3,3,3] — one run burned 16 tool-call turns. The listed items account for 6/9/8 points lost across the three runs; each run's total loss (60 − 51 / 60 − 48 / 60 − 49 = 9/12/11) exceeds that by 3 points, so this list is not a complete per-question attribution. These runs' wall clock was 48/67/52 s; the median (52 s) is inside the ≤1.5× (63 s) gate, but one run at 67 s exceeded the per-run cap. Zero engine runaways in these runs — the loop bug and the restraint gap are different failure modes.

## Table 4 — Cost of the fix and the visible symptom

Condition: same two-node engine, K1 speed probe, 1M YaRN configuration.

| Metric | A: MTP4 + async auto-on | B: MTP4 + `--no-async-scheduling` |
|---|---|---|
| single-stream decode (tok/s) | 52.0 | 52.2 |
| cold prefill (tok/s) | 3034 | 3197 |
| six-stream aggregate (tok/s) | 89.4 | 82.1 (-8%) |
| KV pool tokens | 3,441,324 | 3,455,574 |

Baseline DeepSeek V4 Flash Vision-Exp measured 32.8 / 2129 / 82.8 on the same probe, so the fixed configuration is still +59% single-stream and at parity on six streams.

Visible symptom of a runaway response: the completion degenerates into a repeated token string (for example the same word repeated until max_tokens 4096 is exhausted), taking about 200 s at roughly 20 tok/s; a detector that flags completion > 1500 tokens, or > 60 s, or a run of ≥ 20 identical tokens flags the cases recorded as runaway in the isolation set while the multi-prefill reproducer flags none in this sample (0/40 across four configurations). Raw logs and human labels for a full recall count were not recorded.
