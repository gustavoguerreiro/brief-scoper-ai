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
- **OTel**: OpenTelemetry; vendor-neutral tracing standard.
