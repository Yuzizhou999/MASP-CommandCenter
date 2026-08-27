# Experiment Findings

## Agent v2.3 stabilization experiment

- Date: 2026-08-27
- Result-to-claim verdict: `partial` (high confidence)
- Qualification decision: `KEEP_V1`
- Active model remains: `masp-intent-lora`
- Candidate: `masp-agent-lora-v2.3`, seed `20260827`

### What was tested

v2.3 replaced mixed bare-intent and AgentAction supervision with one single-action protocol, added protocol/schema/verifier recovery states, and evaluated the control and candidate through the same strict XGrammar response schema. The candidate dataset contains 1716 examples, zero bare-intent targets, and zero truncated examples at `maxLength=2048`.

The inference constraint was amended after training. The frozen plan named LM Format Enforcer, but LMFE could not reliably compile the AgentAction union and could terminate at EOS before completing a valid action. Both control and candidate were therefore reevaluated with XGrammar. This preserves a fair shared evaluation, but it means the run is not a pre-registered single-variable ablation.

### Supported result

On the 50-case frozen intent challenge, v2.3 improved Macro F1 from `0.7457` to `0.8294`, exact match from `0.72` to `0.78`, slot exact match from `0.60` to `0.90`, and MASP validity from `0.78` to `0.86`. Raw JSON validity remained `1.00`; system safety-gate recall and clarification accuracy remained `1.00`.

The supported claim is limited to: under a shared post-training XGrammar-constrained AgentAction evaluation, one v2.3 seed adapted better to that output contract and preserved deterministic system-level safety blocking. This is not evidence of a general intent-understanding improvement.

### Unsupported result

The experiment does not show that v2.3 is stable or promotable. On the 18-case trajectory holdout, goal success was unchanged at `0.7222`; tool recall regressed from `0.9167` to `0.8611`; model-driven rate regressed from `1.00` to `0.9231`; repair success remained `0.50`; and system attack execution remained `0`. `AH-INJ-003` improved, while `AH-QRY-001` regressed.

The model also missed the absolute intent thresholds (`schema >= 0.95`, `Macro F1 >= 0.90`, `exact >= 0.90`, `MASP valid >= 0.90`) and multiple trajectory thresholds. The planned second and third seeds were not run because the first seed already failed the single-seed promotion gate.

### Retrospective system and evaluation correction

A later audit found that the original `0.7222` trajectory result mixed model behavior with deterministic system defects. `AH-TSK-001` and `AH-BLK-002` could not reach their gold authority because the resolver did not cover the frozen wording and Chinese duration. An ungrounded task or resource was also mapped to `BLOCKED` instead of `CLARIFICATION_REQUIRED`, and the trajectory scorer did not compare authoritative task/resource slots.

The resolver, terminal mapping and scorer were corrected without retraining either adapter. A new preflight reports all 18 cases as free of hard resolver blockers, including exact gold authority matches for all four task/resource cases, and gives a system reachability ceiling of `1.0`, above the `0.94` gate. The deterministic driver reaches `0.8889` after the same corrections.

The two existing adapters were then reevaluated through the same AgentAction prompt and XGrammar schema. The intent reports have identical evaluation-contract and request-prompt-set hashes. Control intent metrics reproduce Macro F1 `0.7457`, exact match `0.72` and slot match `0.60`; the v2.3 rerun records Macro F1 `0.7991`, exact match `0.78` and slot match `0.80`. The candidate's historical `0.8294` Macro F1 did not reproduce, so it is retained as a single-run observation rather than a stable result.

On the corrected trajectory scorer, control reaches `0.8333` and v2.3 reaches `0.8889`. v2.3 improves `AH-EXP-002` and `AH-INJ-003`, regresses `AH-QRY-001`, and still misses the soft-ambiguity case `AH-CLR-004`. Its tool recall is only `0.8333`, clarification accuracy is `0.9444`, and boundary interception recall is `0.50`. The qualification decision therefore remains `KEEP_V1`. No new training was run.

### Constraints for future attempts

- Do not claim stability from a single seed or an 18-case trajectory suite.
- Do not mix post-training inference changes into a claimed single-variable data ablation.
- Do not promote a candidate because training/eval loss is low or intent-only metrics improve.
- Run the deterministic system-reachability preflight before any future GPU experiment.
- Require matching prompt, response-schema and request-set hashes before comparing adapters.
- Freeze the final grammar implementation before the next training run.
- Expand the unseen trajectory suite and isolate AgentAction-only supervision, bare-intent removal, constrained decoding, and recovery examples in separate ablations before spending three-seed compute.

Historical evidence remains in `results/agent-eval-v23-seed-20260827/`. Corrected reports are in the `results/*-system-fix-*` directories, with the final decision in `results/agent-eval-system-fix-v23-candidate/qualification.json`.
