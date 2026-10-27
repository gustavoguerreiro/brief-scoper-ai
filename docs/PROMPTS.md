# Prompts: versioning, promotion and rollback

Prompts live in `prompts/<name>/vN.md` (front matter + text with `<<placeholders>>`). The file
`prompts/active.json` says which version each prompt uses. `prompts/hashes.json` records the hash of
every version once it exists, and a test fails if a frozen version is edited.

| Prompt | Used by |
|---|---|
| `scope_baseline` | M1 pipeline, no retrieval |
| `scope_grounded` | M2 pipeline, retrieval and citations |
| `judge_pointwise`, `judge_pairwise` | LLM judge instructions |
| `rubric_grounding`, `rubric_actionability` | what pass and fail mean for the judge |

## Rules
1. A version is immutable. To change a prompt, add `vN+1.md`; never edit an existing file.
2. Every eval run stores the id of each prompt it used (`name@version#hash`). Results are only valid for those ids.
3. The judge prompts and rubrics are part of the measuring instrument. Changing one means the human calibration (kappa) no longer applies: relabel or at least recalibrate before trusting judge pass rates again.

## Promote a new version
1. Create `prompts/<name>/v2.md` with a `notes:` line saying what changed and why. Run `uv run python -m scoper.prompts.registry` to freeze its hash.
2. Run the experiment without touching `active.json`, using the environment override:
   ```bash
   SCOPER_PROMPT_SCOPE_GROUNDED=v2 uv run python -m scoper.evals.runner generate --system grounded --split all --out evals/runs/exp_v2.jsonl
   ```
   (PowerShell: `$env:SCOPER_PROMPT_SCOPE_GROUNDED="v2"` first.) The on-demand GitHub workflow `eval-real` does the same from the Actions tab.
3. Gate it on the dev split, then look at the test split once:
   ```bash
   uv run python -m scoper.evals.gate generation --system grounded --split dev  --run evals/runs/exp_v2.jsonl
   uv run python -m scoper.evals.gate generation --system grounded --split test --run evals/runs/exp_v2.jsonl
   ```
   Promote only if both pass, no metric regressed more than `max_regression` against the saved baseline, cost and latency did not get worse without a reason, and (once calibrated) the judge pass rates did not drop.
4. In one pull request: set the version in `active.json`, copy the experiment run to `evals/runs/<system>_all.jsonl`, refresh the baseline with `--save-baseline`, and explain the result in the description. CI will fail the PR if the committed run was produced by different prompt versions than the ones now active.

## Roll back
- **Normal:** revert the promotion commit, or set the previous version back in `active.json`. Old versions are never deleted, so the previous behavior is reproduced exactly.
- **Emergency without a deploy of code:** set `SCOPER_PROMPT_<NAME>=vN` in the environment. This skips the gate, so use it only to return to a version that previously passed.
- After a rollback, regenerate the recorded run for the restored version so the "stale results" check passes again.

## Known limits of this process
- The gate floors in `evals/thresholds.json` were chosen by me with a margin below what I measured on 24 synthetic cases; they are not statistically derived.
- With 16 dev and 8 test cases, one case moves a rate by 4 to 12 points. A pass is evidence of "no obvious breakage", not of improvement.
- Prompts are not canary-released or A/B tested in production in this project.

## DSPy artifacts
`prompts/scope_grounded_dspy/vN.md` holds the DSPy Signature's instructions and is versioned like any other prompt. An optimizer run produces `prompts/dspy/<name>.json` (instructions plus few-shot demos). Treat it as a prompt version: run it on dev, gate it, read test once, and record its hash. Few-shot demos change the cost of every request, so compare cost per case, not only quality.
