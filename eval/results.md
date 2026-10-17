# HABEAS evaluation — Phase 7

Command: `python -m eval.run --alerts all --seed 42 --mode fake --alt-seed 43 --real-smoke`

- Generated: 2026-10-16T20:07:40+00:00
- Seed: 42 · mode: fake · provider: — · model: —
- Graph: `HABEAS_demo_eval_main_s42` (disposable evaluation graph)
- Tool budget: 12 · trace dir: `logs\eval_fake_s42`
- Manifest: `src\seed\ground_truth.json` (seed 42, schema v1)

Evaluated **21** alert(s): 10 suspicious / 11 benign · planted pattern families: 4.

## Outcome mapping

| tier | meaning | counts as |
| --- | --- | --- |
| `supported_for_review` | canonical verdict `suspicion_supported_for_review` | positive risk finding (strict) |
| `suspicious_unconfirmed` | canonical verdict `suspicious_activity` | signal present, escalation not met — neither positive nor benign |
| `benign` | canonical verdict `likely_benign` | benign outcome |
| `abstention` | `insufficient_evidence` / `incomplete_investigation` | inconclusive, visible in the matrix |
| `failed` | harness error or unrecoverable investigation failure | stays in every denominator, reported |

## Headline metrics

| metric | value | rate | sample |
| --- | --- | --- | --- |
| Alert-level recall (strict) | 4/10 | 40.0% | n=10 |
| Alert-level recall (lenient, incl. unconfirmed) | 4/10 | 40.0% | n=10 |
| False-positive rate on benign (strict) | 0/11 | 0.0% | n=11 |
| False-positive rate on benign (lenient) | 0/11 | 0.0% | n=11 |
| Inconclusive rate (all alerts) | 15/21 | 71.4% | n=21 |
| Inconclusive rate (suspicious) | 6/10 | 60.0% | n=10 |
| Inconclusive rate (benign) | 9/11 | 81.8% | n=11 |
| Failure rate | 0/21 | 0.0% | n=21 |
| Investigation completion rate | 21/21 | 100.0% | n=21 |
| Tool-budget exhaustion rate | 0/21 | 0.0% | n=21 |
| Early termination rate | 0/21 | 0.0% | n=21 |
| Pattern-family coverage (strict) | 2/4 | 50.0% | n=4 |
| Pattern-family coverage (>=1 detection) | 2/4 | 50.0% | n=4 |
| Planted signal families observed | 9/10 | 90.0% | n=10 |
| Freeze-eligible under Policy v2 (of attempts) | 1/21 | 4.8% | n=21 |
| Freeze applied (of attempts) | 1/21 | 4.8% | n=21 |
| Verifier pass (of checked claims) | 129/158 | 81.7% | n=158 |

## Confusion matrix (ground truth × outcome tier)

| ground truth | `supported_for_review` | `suspicious_unconfirmed` | `benign` | `abstention` | `failed` | total |
| --- | --- | --- | --- | --- | --- | --- |
| suspicious | 4 | 0 | 0 | 6 | 0 | 10 |
| benign | 0 | 0 | 2 | 9 | 0 | 11 |

## Per-alert results

| alert | family | label | verdict | outcome | action | calls | elapsed_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ALR-0001 | cycle | suspicious | suspicion_supported_for_review | supported_for_review | declined | 7 | 1.114 |
| ALR-0002 | cycle | suspicious | suspicion_supported_for_review | supported_for_review | declined | 8 | 1.133 |
| ALR-0003 | cycle | suspicious | suspicion_supported_for_review | supported_for_review | declined | 8 | 1.098 |
| ALR-0004 | fan_in_out | suspicious | insufficient_evidence | abstention | declined | 8 | 1.19 |
| ALR-0005 | fan_in_out | suspicious | insufficient_evidence | abstention | declined | 8 | 1.226 |
| ALR-0006 | fan_in_out | suspicious | insufficient_evidence | abstention | declined | 10 | 1.229 |
| ALR-0007 | rapid_shared | suspicious | insufficient_evidence | abstention | declined | 7 | 1.083 |
| ALR-0008 | rapid_shared | suspicious | insufficient_evidence | abstention | declined | 8 | 1.115 |
| ALR-0009 | rapid_shared | suspicious | insufficient_evidence | abstention | declined | 8 | 1.092 |
| ALR-0010 | benign | benign | insufficient_evidence | abstention | declined | 7 | 1.357 |
| ALR-0011 | benign | benign | insufficient_evidence | abstention | declined | 8 | 1.372 |
| ALR-0012 | benign | benign | insufficient_evidence | abstention | declined | 7 | 1.652 |
| ALR-0013 | benign | benign | insufficient_evidence | abstention | declined | 7 | 1.326 |
| ALR-0014 | benign | benign | insufficient_evidence | abstention | declined | 8 | 1.657 |
| ALR-0015 | benign | benign | insufficient_evidence | abstention | declined | 7 | 1.198 |
| ALR-0016 | benign | benign | likely_benign | benign | declined | 7 | 1.312 |
| ALR-0017 | benign | benign | likely_benign | benign | declined | 7 | 1.198 |
| ALR-0018 | benign | benign | insufficient_evidence | abstention | declined | 7 | 1.361 |
| ALR-0019 | benign | benign | insufficient_evidence | abstention | declined | 7 | 1.15 |
| ALR-0020 | benign | benign | insufficient_evidence | abstention | declined | 7 | 1.275 |
| ALR-0021 | cycle_rapid | suspicious | suspicion_supported_for_review | supported_for_review | applied ✅ | 7 | 1.156 |

## Per-pattern diagnostics

### cycle

- suspicious alerts: 3 · detected: ALR-0001, ALR-0002, ALR-0003 · missed: —
- detection rate: 3/3 (100.0%)
- evidence tools used: trace_funds×6, alert_context×3, entity_profile×3, detect_cycle×3, fan_in_out×3, hub_scores×3, freeze_account×3, verify_freeze×3, prior_cases×2
- target-relevant evidence: 3/3 (100.0%) · distinct-provenance evidence: 0/3 (0.0%) · freeze-eligible: 0/3 (0.0%)

### cycle_rapid

- suspicious alerts: 1 · detected: ALR-0021 · missed: —
- detection rate: 1/1 (100.0%)
- evidence tools used: trace_funds×2, alert_context×1, entity_profile×1, detect_cycle×1, fan_in_out×1, hub_scores×1, freeze_account×1, verify_freeze×1
- target-relevant evidence: 1/1 (100.0%) · distinct-provenance evidence: 1/1 (100.0%) · freeze-eligible: 1/1 (100.0%)

### fan_in_out

- suspicious alerts: 3 · detected: — · missed: ALR-0004, ALR-0005, ALR-0006
- detection rate: 0/3 (0.0%)
- evidence tools used: trace_funds×6, entity_profile×5, alert_context×3, detect_cycle×3, fan_in_out×3, hub_scores×3, freeze_account×3, verify_freeze×3, prior_cases×2, shared_identifiers×1
- target-relevant evidence: 2/3 (66.7%) · distinct-provenance evidence: 0/3 (0.0%) · freeze-eligible: 0/3 (0.0%)
- miss ALR-0004: shared infrastructure between counterparties was not examined; one or more tool results were truncated, so counts may be incomplete
- miss ALR-0005: shared infrastructure between counterparties was not examined; one or more tool results were truncated, so counts may be incomplete
- miss ALR-0006: one or more tool results were truncated, so counts may be incomplete
  - unresolved hypotheses ALR-0004: shared_identifier_link_warranting_review
  - unresolved hypotheses ALR-0005: shared_identifier_link_warranting_review
  - unresolved hypotheses ALR-0006: —

### rapid_shared

- suspicious alerts: 3 · detected: — · missed: ALR-0007, ALR-0008, ALR-0009
- detection rate: 0/3 (0.0%)
- evidence tools used: trace_funds×6, alert_context×3, entity_profile×3, detect_cycle×3, fan_in_out×3, hub_scores×3, freeze_account×3, verify_freeze×3, prior_cases×2
- target-relevant evidence: 3/3 (100.0%) · distinct-provenance evidence: 0/3 (0.0%) · freeze-eligible: 0/3 (0.0%)
- miss ALR-0007: shared infrastructure between counterparties was not examined; one or more tool results were truncated, so counts may be incomplete
- miss ALR-0008: shared infrastructure between counterparties was not examined; one or more tool results were truncated, so counts may be incomplete
- miss ALR-0009: shared infrastructure between counterparties was not examined; one or more tool results were truncated, so counts may be incomplete
  - unresolved hypotheses ALR-0007: shared_identifier_link_warranting_review
  - unresolved hypotheses ALR-0008: shared_identifier_link_warranting_review
  - unresolved hypotheses ALR-0009: shared_identifier_link_warranting_review

Benign alerts flagged (all): —

## Action and intervention outcomes

| outcome | value | sample |
| --- | --- | --- |
| freeze attempts | 21/21 (100.0%) | n=21 |
| eligible under Policy v2 | 1/21 (4.8%) | n=21 |
| applied (sandbox mutation) | 1/21 (4.8%) | n=21 |
| declined with reasons | 20/21 (95.2%) | n=21 |
| already applied | 0/21 (0.0%) | n=21 |
| in progress | 0/21 (0.0%) | n=21 |
| failed attempts | 0/21 (0.0%) | n=21 |

Policy checks that blocked attempts:

- `benign_explanation_ruled_out`: 12
- `enough_distinct_provenance_families`: 20
- `enough_signal_families`: 17
- `no_material_truncation`: 3
- `signal_families_implicate_target`: 12
- `verdict_supports_review`: 17

## Verifier integrity

- cases verified: 21 · verifier errors: 0
- claims checked: 129/158 (81.7%)
- `reproduced_current_state`: 125
- `verified_against_recorded_evidence`: 0
- `changed_after_action`: 4
- `mismatch`: 0
- `unverifiable`: 29

## Distributions

| group | n | mean | median | p90 | range |
| --- | --- | --- | --- | --- | --- |
| tool calls / alert (all) | 21 | 7.524 | 7.0 | 8.0 | 7.0–10.0 |
| tool calls / alert (suspicious) | 10 | 7.9 | 8.0 | 8.0 | 7.0–10.0 |
| tool calls / alert (benign) | 11 | 7.182 | 7.0 | 8.0 | 7.0–8.0 |
| wall-clock s / alert (all) | 21 | 1.252 | 1.198 | 1.372 | 1.083–1.657 |

Total tool calls: 158. Wall-clock includes MCP server startup and tool execution; FakeLLM adds no network latency.

## One-hop baseline vs bounded traversal (ablation)

Definition: One-hop baseline vs bounded multi-hop traversal over the same MCP boundary and the same 90-day window. The one-hop profile (entity_profile) lists direct counterparties only; the narrow traversal (detect_cycle max_len=3) is limited to <=3-account cycles; the bounded traversal (detect_cycle max_len=4) is inside the validated 2..5 cycle-length range.

| alert | target | 1-hop counterparties (in/out) | ≤3-account cycles (corroborated) | 4-account cycles (corroborated) | contribution |
| --- | --- | --- | --- | --- | --- |
| ALR-0001 | ACC-0387 | 1/1 | 0 (0) | 1 (1) | ring invisible to the one-hop profile and to the <=3-account traversal; found and temporally corroborated by the bounded 4-account traversal |
| ALR-0002 | ACC-0388 | 1/1 | 0 (0) | 1 (1) | ring invisible to the one-hop profile and to the <=3-account traversal; found and temporally corroborated by the bounded 4-account traversal |
| ALR-0003 | ACC-0389 | 1/1 | 0 (0) | 1 (1) | ring invisible to the one-hop profile and to the <=3-account traversal; found and temporally corroborated by the bounded 4-account traversal |
| ALR-0021 | ACC-0407 | 2/2 | 0 (0) | 1 (1) | ring invisible to the one-hop profile and to the <=3-account traversal; found and temporally corroborated by the bounded 4-account traversal |

**Aggregate:** alerts checked=4 · one-hop closures visible=0 · narrow corroborated=0 · bounded corroborated=4

Conclusion: bounded multi-hop traversal temporally corroborated a planted ring on 4/4 circulation alert(s); the one-hop profile alone can express closure on 0/4, and the narrow (<=3 accounts) search likewise 0/4.

## Case-memory comparison (paired)

Method: paired comparison on disposable graphs rebuilt from the same seed per condition; identical warm-up investigation; the only difference is whether the prior_cases tool is part of the agent's interface. The entity_profile tool's pre-existing prior_case_count statistic is unchanged in both conditions (documented limitation).

- `net_call_delta`: calls(memory on) - calls(memory off), summed over pairs
- `prior_lookup_calls`: occurrences of prior_cases in the memory-on repeat tool sequence
- `calls_avoided`: multiset difference of (tool, arguments) steps: steps in the memory-off repeat absent from the memory-on repeat
- `reduction_pct`: (calls_off_total - calls_on_total) / calls_off_total

| pair | warm-up → repeat | rule | calls on | calls off | net Δ | prior lookups | avoided | validated | verdicts agree |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mem1 | ALR-0001 → ALR-0002 | shared_family_members | 8 | 7 | 1 | 1 | 0 | yes | yes |
| mem2 | ALR-0004 → ALR-0006 | same_alerted_account | 10 | 9 | 1 | 1 | 0 | yes | yes |
| mem3 | ALR-0007 → ALR-0009 | same_alerted_account | 8 | 7 | 1 | 1 | 0 | yes | yes |
| mem4 | ALR-0010 → ALR-0011 | same_alerted_account | 8 | 7 | 1 | 1 | 0 | yes | yes |
| mem5 | ALR-0012 → ALR-0014 | same_alerted_account | 8 | 7 | 1 | 1 | 0 | yes | yes |

**Aggregate:** pairs=5 · calls on=42 · calls off=37 · net Δ=5 · prior-case lookups introduced=5 · steps avoided=0 · reduction=-0.1351 · warm-ups equivalent: yes · reused findings validated: 5/5

Conclusion: memory introduced 5 additional prior-case lookup(s) and avoided none of the recorded steps across 5 pair(s); no tool-call saving was measured in this configuration

Wall-clock note: descriptive only: small sample, includes MCP server startup and tool execution; not a performance benchmark (on total 6.427s, off total 6.395s)

## Reproducibility check

Same seed + FakeLLM configuration, graph rebuilt: **identical** (sha256:b4bcc5537052787161db0ffc717b7247a4277ffde86d15b770fda0495180c1bd vs sha256:b4bcc5537052787161db0ffc717b7247a4277ffde86d15b770fda0495180c1bd).

## Alternative-seed validation

Seed 43 (separate denominators, never combined): 21 alert(s), 10 suspicious / 11 benign.

| metric | value | rate | sample |
| --- | --- | --- | --- |
| recall (strict) | 4/10 | 40.0% | n=10 |
| false-positive rate (strict) | 0/11 | 0.0% | n=11 |
| pattern coverage (strict) | 2/4 | 50.0% | n=4 |
| completion rate | 21/21 | 100.0% | n=21 |

## Real-LLM evaluation

- status: **completed**
- provider: groq · model: openai/gpt-oss-120b
- alerts evaluated: 2 (2 suspicious / 0 benign)
- recall (strict): 0/2 (0.0%) · FP rate (strict): 0/0 (n/a) · completion: 0/2 (0.0%)
- ALR-0001: status=ok, verdict=insufficient_evidence, calls=6, elapsed=13.034s, errors=—
- ALR-0002: status=ok, verdict=insufficient_evidence, calls=6, elapsed=65.974s, errors=—

## Failures, misses, and limitations

- truncated evidence on ALR-0001: EV-0007 (bounded row caps; recorded, not hidden)
- truncated evidence on ALR-0002: EV-0008 (bounded row caps; recorded, not hidden)
- truncated evidence on ALR-0003: EV-0008 (bounded row caps; recorded, not hidden)
- MISSED ALR-0004 (fan_in_out): abstention / insufficient_evidence — shared infrastructure between counterparties was not examined; one or more tool results were truncated, so counts may be incomplete
- truncated evidence on ALR-0004: EV-0008 (bounded row caps; recorded, not hidden)
- MISSED ALR-0005 (fan_in_out): abstention / insufficient_evidence — shared infrastructure between counterparties was not examined; one or more tool results were truncated, so counts may be incomplete
- truncated evidence on ALR-0005: EV-0008 (bounded row caps; recorded, not hidden)
- MISSED ALR-0006 (fan_in_out): abstention / insufficient_evidence — one or more tool results were truncated, so counts may be incomplete
- truncated evidence on ALR-0006: EV-0010 (bounded row caps; recorded, not hidden)
- MISSED ALR-0007 (rapid_shared): abstention / insufficient_evidence — shared infrastructure between counterparties was not examined; one or more tool results were truncated, so counts may be incomplete
- truncated evidence on ALR-0007: EV-0007 (bounded row caps; recorded, not hidden)
- MISSED ALR-0008 (rapid_shared): abstention / insufficient_evidence — shared infrastructure between counterparties was not examined; one or more tool results were truncated, so counts may be incomplete
- truncated evidence on ALR-0008: EV-0008 (bounded row caps; recorded, not hidden)
- MISSED ALR-0009 (rapid_shared): abstention / insufficient_evidence — shared infrastructure between counterparties was not examined; one or more tool results were truncated, so counts may be incomplete
- truncated evidence on ALR-0009: EV-0008 (bounded row caps; recorded, not hidden)
- truncated evidence on ALR-0010: EV-0007 (bounded row caps; recorded, not hidden)
- truncated evidence on ALR-0011: EV-0008 (bounded row caps; recorded, not hidden)
- truncated evidence on ALR-0012: EV-0005, EV-0007 (bounded row caps; recorded, not hidden)
- truncated evidence on ALR-0013: EV-0005, EV-0007 (bounded row caps; recorded, not hidden)
- truncated evidence on ALR-0014: EV-0006, EV-0008 (bounded row caps; recorded, not hidden)
- truncated evidence on ALR-0015: EV-0007 (bounded row caps; recorded, not hidden)
- truncated evidence on ALR-0016: EV-0007 (bounded row caps; recorded, not hidden)
- truncated evidence on ALR-0017: EV-0007 (bounded row caps; recorded, not hidden)
- truncated evidence on ALR-0018: EV-0007 (bounded row caps; recorded, not hidden)
- truncated evidence on ALR-0019: EV-0007 (bounded row caps; recorded, not hidden)
- truncated evidence on ALR-0020: EV-0007 (bounded row caps; recorded, not hidden)
- truncated evidence on ALR-0021: EV-0007 (bounded row caps; recorded, not hidden)

Limitations:

- No tool-call saving from case memory was measured in this configuration (see the memory comparison block for the measured delta); nothing is assumed.
- FakeLLM is a deterministic regression baseline: it exercises the real MCP stdio server, Policy v2, the verifier and the production tool-budget enforcement, but it does not demonstrate live model tool selection. That is covered separately by the bounded real-LLM mode/smoke run.
- Wall-clock timings include MCP server startup and are descriptive, not benchmarks.
- The memory comparison is small by construction (pairs derived from the synthetic manifest); its wall-clock deltas are not statistically meaningful, and only tool-call deltas with recorded step evidence are treated as measured savings.
- Alert-level metrics are measured on the synthetic benchmark and do not generalize to production data.
- The agent never receives ground truth; conversely the evaluation can only observe what the agent recorded as evidence. Interpretations are deterministic Python, never LLM-authored.
- Findings are risk indicators for review, not legal conclusions.

Isolation: the agent receives alert ids and graph evidence only; the ground-truth manifest is read exclusively by this evaluation layer, and every evaluation graph is disposable (`HABEAS_demo_eval_*`). Actions run only behind the deterministic Policy v2 checks in the sandbox.

