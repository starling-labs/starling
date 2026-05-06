# Starling: Set-Valued Deep Research Agent

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

Starling is a corpus filter design agent that tackles **set-valued deep research**.

## Artifacts

Released datasets and the entity tagger are available on HuggingFace: [starling-labs/starling-artifacts](https://huggingface.co/collections/starling-labs/starling-artifacts).

- Protein subcellular location
- LD50
- Gene–disease association
- Blood–brain barrier permeation
- Reactions
- Oral bioavailability
- Entity tagger

> **Note:** Automated database construction from an arbitrary corpus is **work-in-progress** — the end-to-end pipeline (entity tagging, normalization, ClickHouse bitmap builds, Milvus indexing) is non-trivial to package, and we're still working on a turnkey path. For now, the search backend assumes a pre-built Milhouse instance.

> **Note on the search backend:** The concrete ClickHouse + Milvus implementation that powered our internal corpus has been **stripped out of this code release**. The agents are written against two ABCs — `VectorSearchBase` (`src/starling/infra/vector_search_base.py`) and `EntityNormalizer` (`src/starling/infra/normalizer_base.py`) — and the modules that previously held the production implementations (`milhouse_vector_search.py`, `ch_normalizer.py`, `ch_client.py`) are now stubs that raise `NotImplementedError`. **The code will not run end-to-end as released.** Bring your own backend by subclassing those two ABCs to point Starling at your own corpus.

> **Coming soon — hosted API & turnkey setup:** A public Starling endpoint is **work-in-progress**. We **will** be updating this repository with (a) a client for the hosted API so you can run the agents against our corpus without standing up your own backend, and (b) a turnkey path for building the search indexes from a corpus of your own. Watch this repo for updates.

## Installation

```bash
uv sync -U
source .venv/bin/activate
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# GCP credentials (Google models, GCS-backed paper storage)
GCP_SERVICE_ACCOUNT_FILE=/path/to/service-account.json
GCP_PROJECT_ID=<project-id>

# Paper storage (full-text corpus)
STARLING_PAPER_ROOT=/path/to/corpus  # or gs://bucket/path

# LLM API key (at least one)
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...

# Milhouse (ClickHouse + Milvus)
CH_HOST=<ip>
CH_PASSWORD=<password>
CH_NATIVE_PORT=<port>
MILVUS_URI=<uri>
MILVUS_TOKEN=<token>

# Optional
LOGFIRE_TOKEN=<logfire-token>
```

## Models

Model strings follow `provider:model_name`. Use `custom:` to point at any OpenAI-compatible endpoint:

- `openrouter:z-ai/glm-4.6`
- `openai:gpt-5`
- `custom:moonshot/Kimi-K2-Thinking@dgx001:32132` — inline endpoint (`http://` assumed)
- `custom:moonshot/Kimi-K2-Thinking` — resolved via `CUSTOM_ENDPOINTS_JSON='{"moonshot/Kimi-K2-Thinking":"http://dgx001:32132"}'`

## Usage

### Design a retrieval filter

```bash
starling mode=run \
  task='"Find all molecules mentioned in PubMed for which BBB permeability status (permeable or impermeable) has been reported."' \
  models.main="openai-responses:gpt-5.2" \
  models.sub="custom:openai/gpt-oss-120b@localhost:9000"
```

`models.sub` covers the extractor, validator, and investigator agents; each can also be set individually.

Wrap the task in single-then-double quotes (`'"..."'`) — Hydra quirk.

Outputs (under `./runs/<run_name>/`):
- `retrieval_spec.json` — the designed filter
- `agent_run.json` — full agent conversation log

Configuration is a typed Hydra structured config in `src/starling/cli/conf/config.py` with defaults in `config.yaml`. Override anything via `key=value`.

### Run extraction from a saved filter

```bash
starling mode=extract \
  spec_file="./runs/<run_name>/retrieval_spec.json" \
  task="Extract BBB permeability data including compound name, Kp value, and experimental method" \
  extract.limit=1000 \
  extract.parallelism=20 \
  extract.window_parallelism=20 \
  outputs.extraction="./runs/<run_name>/bbb_extractions.jsonl"
```

Useful overrides:
- `extract.parallelism` — concurrent papers
- `extract.window_parallelism` — concurrent extraction windows across all in-flight papers
- `extract.window_paragraphs`
- `extract.pmids_file` — JSON array of PMIDs; bypasses retrieval from `spec_file`

### Resume an interrupted extraction

```bash
starling mode=extract \
  extract.resume_dir="./runs/<run_name>" \
  extract.parallelism=20 \
  extract.window_parallelism=20
```

Resume reuses `pmids_to_process.json` and `extraction_guidance.json`, skips PMIDs already in `extractions.jsonl`, and appends new results. No need to pass `spec_file` or `task`.

## Output formats

### RetrievalSpec

```json
{
  "filters": [
    {
      "entity_groups": [["SmallMolecule"], ["blood-brain barrier"]],
      "expand_entities": true,
      "semantic_query": "this molecule crosses the blood-brain barrier",
      "estimated_count": 12847,
      "precision_estimate": 0.82
    }
  ]
}
```

`entity_groups` is CNF: outer = AND, inner = OR. Prefix terms with `~` for negation.

### Extraction outputs

```json
// extractions.jsonl (one line per paper)
{
  "pmid": "12345678",
  "title": "Brain penetration of novel compounds...",
  "status": "success",
  "extractions": [
    {
      "extraction_id": "ext_1",
      "support": {
        "paragraph_idx": 42,
        "support_text": "Compound 4a showed a Kp,brain of 0.85, indicating substantial brain penetration in the reported assay."
      },
      "extracted": {"compound_name": "Compound 4a"},
      "confidence": 0.92
    }
  ]
}
```

`support.support_text` is a readable rendering of the supporting evidence — close to the paper text but not required to be a verbatim quote.

The run directory also contains `pmids_to_process.json` (PMID list) and `extraction_guidance.json` (task instantiation + extracted schema). Use `ExtractionBatchResult.load(...)` to reconstruct the full batch.

## Architecture

```
starling mode=run
    │
    ├── CorpusFilterAgent (main loop)
    │   ├── Tools: normalize, search, validate, probe, extract_sample, investigate
    │   ├── ValidatorAgent (judges paper relevance)
    │   ├── InvestigatorAgent (deep paper analysis)
    │   └── ExtractorAgent (structured data extraction)
    │
    └── Output: RetrievalSpec

starling mode=extract
    │
    ├── Load RetrievalSpec
    ├── Execute retrieval → PMIDs
    └── Parallel extraction (ExtractorAgent × N)
        └── Output: ExtractionBatchResult
```
