# Multi-Model Orchestration Engine

A model-agnostic, local-first foundation for coordinating models, agents, tools, and reliable task execution.

This is a new project with its own repository and architecture. The first milestone intentionally contains only the provider boundary, configuration, and a local Ollama adapter. Orchestration, routing, task state, tools, agents, and evaluation will be added incrementally behind these contracts.

## Current milestone

- Typed, provider-neutral chat contracts in `src/mmo_engine/models/base.py`.
- Environment-backed local configuration in `src/mmo_engine/config.py`.
- Ollama HTTP adapter in `src/mmo_engine/models/ollama.py`.
- Provider errors are normalized and marked retryable where appropriate.
- Installed model discovery does not infer capabilities from model names.
- No model is downloaded automatically.
- Unit tests use mocked HTTP responses and do not require Ollama to be running.

## Local setup

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

The default configuration points to a local Ollama service at `http://127.0.0.1:11434` and uses `qwen2.5:7b` as an explicit default candidate. The provider does not assume that model is installed; call `list_models()` and route only to models that are actually available.

To inspect the local environment without downloading anything:

```bash
ollama list
ollama serve
```

The daemon must be running only for live inference. Tests remain offline and mocked.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MMO_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama HTTP endpoint |
| `MMO_DEFAULT_MODEL` | `qwen2.5:7b` | Initial model candidate |
| `MMO_REQUEST_TIMEOUT_SECONDS` | `60` | Per-request timeout |
| `MMO_LOCAL_ONLY` | `true` | Keeps the first milestone local-only |

## Architecture direction

The intended dependency direction is:

```text
interfaces → orchestration → routing / agents / tools
                           → model contracts → provider adapters
                           → storage / evaluation
```

Provider-specific HTTP details stay inside provider adapters. The future orchestrator will consume `ModelProvider`, `ChatRequest`, and `ChatResponse` rather than importing Ollama directly.

## Next step

Add a small model registry and deterministic router that discovers available Ollama models, accepts explicit capability metadata, and returns a selected model plus a routing reason. No LLM-based routing or multi-agent execution is needed until that foundation is tested.

