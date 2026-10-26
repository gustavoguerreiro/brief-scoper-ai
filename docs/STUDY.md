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
- **OTel**: OpenTelemetry; vendor-neutral tracing standard.
