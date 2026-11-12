# 0006. Catalog enrichment as a queued job, schema-validated and grammar-constrained

- Status: accepted
- Date: 2026-11-12

## Context

Product catalogs arrive from suppliers and marketplace feeds with titles in
capitals, SKU codes, promotional noise, and descriptions full of HTML. A
catalog needs the same facts as structured fields: product type, brand,
material, dimensions and a clean title. This is extraction, not
conversation: there is one right answer per product, it has a fixed shape,
and a wrong or malformed answer must never reach the catalog unnoticed.
Volumes are batch-shaped (a feed of thousands of products), so latency per
product matters less than throughput, cost and the share that comes out
usable.

## Decision

**A job type on the existing queue.** `POST /v1/jobs` with `url:
/v1/catalog/enrichments` and `body: {title, description}`. It reuses
everything the completion jobs already have:

- RabbitMQ quorum queues with per-delay retry queues and a dead-letter
  queue;
- Redis job records and idempotency keys;
- KEDA scaling the workers from zero on queue length;
- traces from submission to the last attempt.

The worker cleans the input deterministically (decodes entities, turns HTML
tags into line breaks), then asks for one JSON object at temperature 0.

**The answer is validated, never trusted.** Every answer is parsed and
checked against `app/catalog/product.schema.json` (JSON Schema 2020-12,
with `jsonschema`):

- a category from the taxonomy enum;
- brand and material as strings;
- dimensions as positive numbers with a unit enum;
- a title of bounded length;
- no other fields.

An answer that fails validation is a failed attempt (422). The job is
retried through the retry queues, and the validation error is added to the
next prompt: with greedy decoding the same prompt would produce the same
invalid answer. After the last attempt the job fails, and the message goes
to the dead-letter queue, where products the model cannot handle collect
for inspection. Tokens and generation time are summed over every attempt in
the job's `usage`, so a product's cost includes its rejected answers.

**The output is constrained to the schema while it is generated.** Where the
backend allows it, the schema is enforced during decoding, not only checked
afterwards:

- llama.cpp compiles it into a GBNF grammar that masks every token the
  schema does not allow;
- OpenAI-compatible chat backends receive it as a strict `json_schema`
  `response_format`;
- Anthropic and the completions API rely on the prompt.

The post-hoc validation stays in every case.

**Measured, not assumed.** `scripts/build_catalog_dataset.py` builds 200
labelled rows, 20 per product type, from Amazon Berkeley Objects listings
(CC BY 4.0, archive pinned by SHA-256, rebuilt byte for byte):

- **Inputs** are each listing's own name and bullet points, corrupted
  deterministically.
- **Labels** are its structured fields.

Rows are dropped when:

- the brand is not in the name;
- the material is not stated in the listing's text, or only echoes the
  brand;
- any label fails the schema.

`scripts/eval_catalog.py` submits every row as a job through the API and
the queue, and grades each field:

| Field | Correct when |
|-------|--------------|
| category | exact match |
| brand, material | match after normalisation |
| dimensions | within 2% after conversion to cm |
| title | token F1 ≥ 0.8 |

CI runs it on every build against the built image, with three workers, and
gates on it.

Qwen2.5-0.5B Q2_K, 3 workers of 1 CPU each, same 200 products:

| | Prompt only | Grammar-constrained |
|---|---|---|
| Valid record (after retries) | 46.0% | **100%** |
| Valid on the first attempt | 41.5% | 100% |
| Mean attempts | 2.14 | 1.00 |
| Category / brand / material | 32.5% / 23.5% / 41.5% | 75.5% / 44.0% / 91.5% |
| Dimensions / title | 38.0% / 11.0% | 63.0% / 34.5% |
| Mean field accuracy | 29.3% | **61.7%** |
| All five fields right | 2.5% | 4.5% |
| Generation time p50 / p95 | 18.2 s / 31.9 s | 6.8 s / 8.7 s |
| Tokens per product (prompt + completion) | 856 + 262 | 381 + 85 |
| Compute cost per product (1 vCPU at $0.0336/h) | $0.000157 | $0.000064 |
| Throughput | 10.6 products/min | 26.2 products/min |

Without constraint, 78 of the 108 failures were not JSON: answers that
rambled to the token limit. Another 28 added fields the schema forbids.
Retries with feedback fixed only 7 products, because a 0.5B model rarely
corrects itself. The grammar removed the structural failures. It also
improved the content, because the model no longer spends tokens and
attention on format, and generation stops when the object closes. That is
why time and cost fell by more than half.

The CI gate is set at ≥ 95% valid and ≥ 0.55 mean field accuracy, a little
below the constrained baseline (`evals/baselines/`).

## Alternatives considered

- **A synchronous endpoint.** This is simpler for one product, but a feed of
  thousands would hold connections for minutes, could not be retried or
  dead-lettered per product, and could not scale workers on backlog. The
  queue already provides all three.
- **A separate queue or service for catalog work.** It would isolate
  workloads, but it duplicates the topology, scaler, alerts and runbooks.
  The cost is shared capacity with completion jobs, which the KEDA scaler
  already sizes.
- **Validation without constrained decoding.** This was the first
  version, and the first baseline measured it: 46% valid. It is the right
  design for a backend that cannot enforce a schema, and it remains the
  path for those.
- **Constrained decoding without validation.** Not every backend enforces
  the schema, and a grammar does not check every keyword
  (`exclusiveMinimum`, string lengths). Validation costs microseconds.
- **Pydantic models instead of JSON Schema.** These are equivalent inside
  Python. A JSON Schema file is language-neutral: the grammar compiler,
  OpenAI's structured outputs and any consumer of the catalog read the same
  document.
- **A larger model or fine-tuning.** These would raise accuracy, especially
  for brand and title, where the 0.5B model loses most. They are a change
  of `VORTEX_MODEL_PATH` measured by the same evaluation, not a design
  decision. The CI model is the one that fits a free runner.
- **Other public data.** Open Food Facts has no materials or dimensions.
  Icecat's open catalog requires an account. Synthetic products would
  measure the generator, not the task. ABO has real listings with
  structured fields under a permissive licence. Its English listings are
  mostly home goods, and its brands are mostly Amazon's own (Rivet, Stone &
  Beam), which makes brand extraction easier than on a mixed catalog.
- **An LLM as judge of correctness.** It is useful for free text, but
  these fields have checkable answers, and a judge would add cost and its
  own error rate to the measurement.

## Consequences

- Every enriched record in the catalog matches the schema, whatever the
  backend. Products the model cannot handle end in the dead-letter queue,
  which raises `VortexJobsDeadLettered`. The runbook covers re-submitting
  them after a prompt or model change.
- Accuracy is measured on every build. A prompt or model change that
  improves one field and breaks another shows up per field and per
  category in the report.
- Cost per product is measured as tokens and generation seconds. A price
  turns it into money: a vCPU-hour price for local models, or a per-token
  price for hosted ones (`--input-price`, `--output-price`).
- The taxonomy is the schema's enum. A new product type is a schema change,
  plus labelled rows for it.
