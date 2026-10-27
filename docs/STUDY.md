# STUDY NOTES

## M1. Baseline: structured output, retries, cost accounting

**Concept (plain language).** A language model returns free text, but a backend needs data it can trust. So I describe the shape I want (a schema), ask the model for JSON in that shape, and validate the reply in code. If it is invalid, I send the error back and ask for a fix (a repair loop), within a limit. Calls can fail for temporary reasons (rate limits, server errors), so I retry those with growing waits, but never retry permanent errors like a bad key. Every call goes through one wrapper that records tokens, cost and latency, so cost is a number per request, not a surprise at month end. A per-request cost cap stops retry storms. Because everything passes through one wrapper, tests swap in a fake model and run offline.

**Trade-offs / when NOT to use.**
- Prompt-asks-for-JSON plus validation works on any provider, but native structured-output modes can reduce invalid replies. I did not measure the difference. Use the native mode when the provider offers it and you only target that provider.
- Repair loops cost extra calls; cap them. Do not loop on a model that fails for semantic reasons.
- Retrying inside the SDK and in my code would double-retry, so I turned the SDK's off (`max_retries=0`) to see every attempt in the log.
- A hand-written wrapper is fine for one or two providers; with many, consider a gateway library.

**How I built it (system-design explanation).**
- *Structured output:* I define a Pydantic schema, put its JSON schema in the prompt, and validate every reply in code. If validation fails, I send the error back for up to two repairs and cap the cost per request. The client brief goes in a delimited block that the prompt marks as untrusted data. The repair path is tested offline with a fake model that returns bad JSON first. I have not yet measured how often a real model needs repairs.
- *Cost and failures:* one client wrapper logs tokens, cost and latency for every call. Retries use exponential backoff with jitter, only for rate limits, 5xx errors and network errors, and each attempt is logged, so failed attempts are visible. A per-request cap, 5 cents by default, stops the repair loop when spend gets too high. Prices live in a config file, which I flagged as unverified.
- *Testing:* I hide the provider behind a small interface and use a deterministic FakeModel that returns scripted replies or raises errors. That lets me test retries, repair, budget caps and API status codes offline in about a second. Those tests prove my logic, not model quality; the evals in M3 cover quality.

**Honest limits.** No real model call has been measured yet (no key in this environment), so real cost, latency and repair rate are unknown; the estimate is ungrounded until retrieval exists in M2.

---

## M2. RAG over past briefs: hybrid retrieval and verified citations

**Concept (plain language).** Instead of letting the model guess an estimate from its general knowledge, I first look up similar past projects and show them to the model, so its answer can point at evidence. Past briefs are cut into chunks, one or more per section. Each chunk is turned into an embedding, a list of numbers where similar meaning lands close together, and stored in Postgres with the pgvector extension. At query time I run two searches: a keyword search (BM25, which rewards rare matching words) and a vector search (which catches paraphrases). I merge both rankings with Reciprocal Rank Fusion, which only uses positions, so the two score scales never need to be compared. Filters on metadata (industry, type, visibility) are applied before ranking. The model must cite chunk ids with word-for-word quotes, and my code checks every citation, because a model can fabricate a citation as easily as a fact.

**Trade-offs / when NOT to use.**
- Hybrid search costs two pipelines and a fusion step. With a tiny corpus, either one alone may do as well. I will measure that in M3 instead of assuming hybrid wins.
- Small chunks match precisely but lose context; big chunks keep context but dilute the match. I chunk by section and prefix each chunk with the brief title and section name as a cheap compromise.
- Verifying that a quote exists proves nothing about whether it supports the claim. That needs a judge, and a judge has its own biases.
- RAG is the wrong tool when the knowledge is small enough to fit in the prompt, when answers need reasoning over the whole corpus, or when stale documents are worse than no documents.
- pgvector keeps vectors next to metadata and filters in one transactional store; a dedicated vector database only pays off at a scale this project does not have.

**How I built it (system-design explanation).**
- *Ingestion:* each brief is a markdown file with YAML front matter (industry, type, budget band, outcome, actual hours and cost, visibility) and four sections: context, scope, constraints, outcome. An import script validates every file with Pydantic, reports readable errors, skips duplicate ids, warns when a delivered project has no actuals (an estimate cannot be grounded without them), and has a dry-run mode that needs no database or key. Chunk ids look like `b001:outcome:0`, so a citation can be checked by eye.
- *Retrieval:* BM25 runs in-process with `rank_bm25` over the filtered chunks, vector search runs in Postgres with cosine distance, and RRF merges the two lists. Filters run in SQL before ranking. A permission-style filter hides private briefs by default, and an exclude list lets evaluation run leave-one-out so a brief cannot retrieve itself. Embeddings sit behind the same kind of seam as the chat model, with a deterministic fake for tests.
- *Generation:* the retrieved chunks go into the prompt as data, with their project actuals in the header. The output schema attaches citations to each risk and to the estimate. Code checks that each chunk id was retrieved, that the quote appears word for word in that chunk, and that an estimate range stays within a band around the cited projects' real hours or cost. A failure triggers a repair call that quotes the problem. If repairs run out, the unverifiable claim is dropped, flagged in the response, and the confidence is lowered, instead of failing the whole request. If nothing relevant is retrieved, the model must return a null estimate and code removes any invented one.

**Honest limits.** I have not measured retrieval quality yet (no real briefs, no embedding key in use), so I cannot claim hybrid beats either search alone. The tests use a fake embedder that only captures shared words, so they prove the plumbing and not semantic quality. The estimate band (0.5x to 2x) is an arbitrary starting value. Citation checks prove a quote exists, not that it supports the claim.

---

## M3. Evals first: retrieval metrics, code checks and a calibrated LLM judge

**Concept (plain language).** An eval is a fixed set of test cases plus a way to score the system on them, so that "it seems better" becomes a number you can compare before and after a change. There are levels, and each one answers a different question. First, retrieval metrics ask whether the right past projects were found, with no model involved: hit@k is the share of cases where a correct project appears in the top k results, and MRR averages 1 divided by the position of the first correct result. Second, code checks ask hard yes/no questions that need no judgment: is the output valid, do the citations exist, did the system refuse when it should. Third, for open-ended quality, such as whether the risks are specific and the claims are supported, I use a language model as a judge, with one criterion per call and a written rubric. A judge is itself a measuring instrument that can be wrong, so I compare it with my own labels and report agreement, using Cohen's kappa, which discounts the agreement you would get by chance. I also swap answer order to check for position bias.

**Trade-offs / when NOT to use.**
- Code checks are exact and free but only see what you thought to check; mine passed 24 out of 24 and therefore could not rank anything. They are a floor, not a quality score.
- An LLM judge scales where humans do not, but it has known biases: it may favor the first or last answer (position), longer answers (verbosity) or its own model family (self-preference), and it can be lenient. Use it only after calibration, and never as the only signal.
- Pairwise comparison is easier for a judge than absolute scoring, but it only ranks two systems against each other and costs two calls per case because of the order swap. Pointwise gives a pass or fail per output, which is what I can calibrate against human labels.
- With about 20 human labels, kappa has a very wide confidence interval, so it can only show gross disagreement. A kappa also breaks down when almost every label is the same class.
- Do not build a judge when a code check can answer the question, and do not tune a prompt on the same cases you report. I split the cases into dev for tuning and test for reporting.
- Evals written by the same author or model as the data look easier than real life.

**How I built it (system-design explanation).**
- *Eval set:* 24 cases stored as JSON lines, each with the new brief, the past briefs a person would call relevant, whether the correct answer is to abstain, an optional injection canary string, and a dev or test split. They cover typical briefs, paraphrases of corpus briefs, prompt-injection attempts, vague briefs and out-of-domain briefs. A second set of 10 hard paraphrases deliberately avoids the corpus vocabulary, for retrieval only, because the first set was too easy to separate the methods.
- *Retrieval evals:* the harness ranks briefs from the fused chunk list and computes hit@1, 3 and 5 and MRR for BM25, vector, hybrid, and hybrid with at most two chunks per brief. It also reports how many distinct briefs sit in the top five chunks. Query embeddings are cached so comparing methods does not repeat paid calls.
- *Generation checks:* plain code verifies schema validity, citation validity (nothing had to be dropped), required fields, correct abstention (an estimate exists exactly when the case expects one) and injection resistance. The injection check flags a hijacked output, not a mere quote of the injected text. I found that false positive by reading outputs, fixed the check, and recomputed the stored results at no cost.
- *Judge:* a stronger model tier than the generator, to reduce self-preference. It receives the brief, the sources and the output as data, a rubric for one criterion, and returns the reason before the verdict. Grounding asks whether every claim about past projects is supported by the sources, which covers the free-text summary that citation checks cannot see. Actionability asks whether the risks and questions are specific to this brief. Pairwise runs both orders, maps verdicts back to systems, and reports consistency and how often the judge picked the same slot both times.
- *Calibration:* a terminal tool shows me each sampled output blind to the judge and records pass or fail per criterion. A report joins my labels with the judge verdicts and prints agreement, kappa with a bootstrap interval, the confusion counts and each side's pass rate.
- *Cost control:* every real run has a spend cap, and results are written to files so a failed or interrupted run can be inspected without paying again.

**Measured so far.** On the main retrieval set every method scored 1.000, which only shows the set is too easy. On the hard paraphrase set (10 cases) BM25 reached MRR 0.808 while vector and hybrid reached 1.000, so vector alone matched hybrid. The grounded pipeline passed all code checks on 24 cases with no repairs, at a median of about 2 cents and 12 seconds per case; the baseline never produced an estimate by design. The pairwise judge picked the second slot far more often than the first (23 against 11 verdicts) and agreed with itself across an order swap in only 13 of 24 cases. The pointwise judge failed 5 of 24 grounding checks and none of the 72 actionability checks. Against my 40 labels the agreement was 90 percent but kappa was 0.00, because I marked every item as pass; that makes kappa meaningless, so the calibration is inconclusive.

**Honest limits.** Everything runs on synthetic data written by one author, the eval cases included, so retrieval looks easier than it would on real briefs. The hard set has 10 cases, so one case moves a metric by 10 points. My own labels contain no failures, so kappa cannot say whether the judge detects bad outputs; the four items where the judge disagreed all went the same way (it was stricter than me on causal claims the sources do not state). Actionability passes everything, for the judge and for me, so it does not discriminate between systems. When I revised three labels after reading the judge's reasons, agreement rose to 95 percent and kappa to 0.83, but that number is contaminated: changing labels after seeing the judge's arguments anchors the human to the judge, so it proves nothing about calibration. A valid check needs independent labels with both classes present. To measure the judge without relying on labels, I planted known defects into 20 good outputs: the judge failed all 40 planted defects (wrong number, invented claim, generic questions) and passed 16 of 20 grounding and all 20 actionability controls. That shows sensitivity to obvious defects, not to subtle ones, so I use the judge as a regression alarm and not as proof of quality. The baseline comparison on abstention is close to trivial because its prompt forbids estimates. The judge shares an API provider and general training lineage with the generator, so some self-preference can remain even across model tiers.

---

## M4. Prompt versioning and a quality gate in CI

**Concept (plain language).** A prompt is code that changes behavior, so it should be treated like code: stored in version control, reviewed, tested and released on purpose. Each change becomes a new immutable version, and one small file says which version is live. A quality gate is an automated check that turns eval results into pass or fail, so a bad change cannot slip through because someone forgot to look. Two kinds of rules are useful. A floor says a metric must not fall below a minimum, and a regression rule says it must not drop by more than a margin compared with the last accepted result. Cheap, deterministic checks run on every change in CI; expensive checks that call real models run only when someone asks. Promoting a version means changing the pointer after the gate passes, and rolling back means pointing at the previous version, which still exists unchanged.

**Trade-offs / when NOT to use.**
- Prompts in Git give diffs, review and easy rollback but need a code deploy to change; a database or prompt-management service allows live edits and A/B tests at the cost of another system and weaker review. For a small team, files win.
- Floors catch disasters, regressions catch slow decay. Floors set too high make the gate flaky and teach people to ignore it; set too low they pass broken changes. Mine are chosen with a margin, not derived.
- A gate on a small eval set is a smoke alarm, not a measurement. With 16 dev cases one case moves a rate by about 6 points, so a pass means "nothing obviously broke".
- Gating on the judge is only valid after the judge is calibrated; otherwise you gate on noise. Uncalibrated checks should be loudly skipped, not silently passed.
- Do not run real-model evals on every commit: they cost money, take minutes and vary from run to run. Keep per-commit CI offline and deterministic.
- Do not tune thresholds or prompts on the data you report; use dev to decide and test to confirm.

**How I built it (system-design explanation).**
- *Prompt registry:* every prompt is a versioned file with placeholders, and an active pointer picks the version. A hash file freezes each version, and a test fails if a frozen version is edited, so history cannot be rewritten by accident. An environment variable can override a version for an experiment or an emergency rollback.
- *Provenance:* every eval run row stores the exact prompt ids it used. The gate refuses results produced by a different prompt than the one currently active, or by an unrecorded one, which forces a rerun whenever a prompt changes.
- *Gate:* a small program reads the results, applies floors and the regression check against a saved baseline, prints each finding and exits non-zero on any failure. Skipped checks are reported, and a strict mode makes them fail.
- *CI:* on every push and pull request the pipeline lints, runs all offline tests, runs the retrieval gate with fake embeddings, and gates the recorded real-model run. The tests include the whole eval pipeline with a well-behaved scripted generator, which must pass the gate, and a generator that fabricates citations, which must fail it. A gate that cannot fail proves nothing, so the failing case is part of the suite. A separate manual workflow runs the real model with a spend cap against a Postgres service and uploads the artifacts.
- *Process:* a written procedure covers adding a version, testing it by override on dev, confirming on test, promoting through one pull request, and rolling back by revert or by pointing at the previous version.

**Measured so far.** The offline retrieval gate and the dev-split gate on the recorded run both pass, with the floors shown in `RESULTS.md`. The pipeline tests show the gate passing for a good generator and failing for a fabricating one.

**Honest limits.** The GitHub workflows have never run on GitHub; I only checked that the YAML parses and ran the same commands locally. The floors are my choices on 24 synthetic cases. The judge is only partly validated (crude planted defects caught, human calibration inconclusive), so its gate floors are a regression alarm. Prompts are not tested in production traffic. Changing a judge prompt invalidates the human calibration, and the process says so but nothing enforces it automatically.

---

## M5. DSPy: prompts as programs, and optimizers

**Concept (plain language).** DSPy lets you describe a language-model step as a program: a signature says what goes in and what comes out (for example a brief and some sources in, a typed scoping result out), and a module says how steps are chained. The framework turns that into the actual prompt. The key idea is that a prompt becomes something a program can improve. An optimizer takes your module, a small set of training examples and a metric, runs the program on the examples, keeps the runs that score well, and puts them into the prompt as worked examples (bootstrapping few-shot demos). More advanced optimizers also rewrite the instructions and search over many candidates. This only works when you have a metric that separates good from bad outputs and examples that represent real traffic. The optimizer cannot be better than the metric it is given.

**Trade-offs / when NOT to use.**
- DSPy removes hand-tuned prompt strings and gives a clean place to plug in an optimizer, but adds a dependency, hides the final prompt behind adapters, and its LM plug-in interface changed within a minor version (the custom-LM subclass I use is deprecated in 3.4 and removed in 3.5).
- Few-shot demos raise the cost of every request: in my run the optimized program cost 1.82 times as much per case. Optimization is paid once, but the demo tokens are paid forever.
- Optimizers need headroom: if the metric is already near its maximum, there is nothing to optimize. Mine was at the ceiling before I started.
- Optimizers can overfit a small training set and the metric: they learn what the metric rewards. Use a held-out split, a metric the optimizer did not train against, and a noise estimate.
- Do not use an optimizer when a prompt edit guided by reading failures would do, when you cannot afford the evaluation calls, or when the eval set is too small to tell a real gain from run-to-run variation.

**How I built it (system-design explanation).**
- *Rewrite:* the grounded generation step is a DSPy signature and module with a main predictor and a repair predictor. Retrieval, the deterministic citation checks, the repair loop and the dropping of unverifiable claims stay in my code, so DSPy can change the wording and the demos but cannot weaken the guarantees. The old pipeline stays as the production default.
- *One model client:* DSPy would normally call the provider through its own library. I wrote a thin language-model adapter that forwards every call to my single client wrapper, so costs, retries and the fake model for tests all still work, and added a hard spend budget that stops an optimizer run.
- *Optimizer and metric:* BootstrapFewShot, trained on the 16 development cases only. The metric gives half a point for passing all code checks and half for a grounding pass from a small, cheap judge. I validated that cheap judge first with the planted-defect test, then judged the held-out results with the stronger judge, so the optimizer was not graded by its own metric model.
- *Evaluation:* before and after on the 8 held-out cases, plus a rerun of the unchanged production pipeline to measure noise, plus cost and latency per case.

**Measured so far.** The optimizer kept 3 demos after trying only 3 of the 16 training examples, because it stops once it has enough passing traces; it did no search. The optimized program scored the same as the unoptimized one on every check and 8 of 8 on the strong judge's grounding verdicts, but the unoptimized DSPy program was already 8 of 8. The hand-written production pipeline scored 5 of 8 and then 7 of 8 on two identical runs. The optimized program was 1.82 times as expensive per request. So the optimizer's benefit is not measurable here, and its cost is.

**Honest limits.** Eight held-out cases and single runs cannot show a real difference; the noise between two identical runs was larger than any gap. The metric was at its ceiling, so there was no room to improve. BootstrapFewShot is the simplest optimizer and did no search; I did not run MIPRO or random search, so I cannot say what they would do. There is no train-versus-test comparison, so I cannot quantify overfitting. The DSPy rewrite may be a better prompt than my hand-written one on the grounding judge (8 of 8 against 5 and 7 of 8), but that is not established either. The adapter relies on a deprecated DSPy interface.

---

## Glossary
- **RAG**: Retrieval-Augmented Generation; fetch relevant documents, then have the model answer using them.
- **BM25**: keyword ranking function based on term frequency and rarity.
- **RRF**: Reciprocal Rank Fusion; merges ranked lists by summing 1/(k+rank).
- **MRR**: Mean Reciprocal Rank; average of 1/rank of the first correct result.
- **hit@k**: share of queries with a correct result in the top k.
- **Cohen's kappa**: agreement between two raters corrected for chance.
- **DSPy**: framework that treats prompts as programs and optimizes them against a metric.
- **ReAct**: agent pattern alternating reasoning and tool actions.
- **SFT**: Supervised Fine-Tuning on input/output pairs.
- **RLHF**: Reinforcement Learning from Human Feedback.
- **DPO**: Direct Preference Optimization; trains on preferred vs rejected pairs without a reward model.
- **GRPO**: Group Relative Policy Optimization; RL method comparing a group of sampled answers.
- **LoRA**: Low-Rank Adaptation; trains small add-on matrices instead of all weights.
- **QLoRA**: LoRA on a quantized (low-precision) base model to save memory.
- **PEFT**: Parameter-Efficient Fine-Tuning; umbrella for LoRA-like methods.
- **FTS**: Full-Text Search (Postgres).
- **HNSW**: graph index for approximate nearest-neighbor vector search.
- **HITL**: Human-in-the-loop.
- **Embedding**: a vector of numbers representing a text so that similar meanings are close.
- **pgvector**: Postgres extension that stores embeddings and searches them by distance.
- **Cosine similarity**: closeness of two vectors by angle, ignoring length.
- **Chunk**: a piece of a document, the unit that is embedded and retrieved.
- **Leave-one-out**: evaluating a case with its own source removed from the corpus.
- **Eval set**: fixed test cases plus scoring rules, used to compare versions of the system.
- **LLM-as-judge**: using a language model to grade another model's output against a rubric.
- **Rubric**: written criteria that tell the judge exactly what pass and fail mean.
- **Pointwise / pairwise**: grading one output on its own, versus choosing the better of two.
- **Position bias**: a judge favoring an answer because of its slot (first or last), not its content.
- **Self-preference**: a judge favoring outputs that resemble its own model family's style.
- **Calibration**: checking a judge against human labels before trusting it.
- **Ceiling effect**: a test so easy that every system scores near the maximum, so it cannot rank them.
- **Canary**: a marker string planted in a test input to detect that an injected instruction was followed.
- **CI/CD**: Continuous Integration and Delivery; automatic checks and releases on every change.
- **Quality gate**: an automated pass/fail check on eval results that blocks a change when it fails.
- **Floor / regression check**: a minimum allowed metric value, versus a limit on how much it may drop from a baseline.
- **Promotion / rollback**: making a new prompt version live, versus returning to the previous one.
- **Provenance**: the record of exactly which prompt and model produced a result.
- **Signature (DSPy)**: a declaration of a step's inputs and outputs, from which DSPy builds the prompt.
- **Module (DSPy)**: a program that chains predictors and ordinary code.
- **Optimizer / teleprompter (DSPy)**: a procedure that improves a module's prompts or demos against a metric.
- **BootstrapFewShot**: runs the program on training examples and keeps passing runs as few-shot demos.
- **MIPRO**: an optimizer that proposes and searches instruction and demo candidates (not run here).
- **Few-shot demo**: a worked input-output example placed in the prompt.
- **Overfitting (optimizer)**: improving on the training examples and metric without improving on new data.
- **OTel**: OpenTelemetry; vendor-neutral tracing standard.
