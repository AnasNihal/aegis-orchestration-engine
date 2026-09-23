# Aegis Orchestration Engine

A model-agnostic, local-first foundation for coordinating models, agents, tools, and reliable task execution.

This project has its own repository and architecture. It is being built incrementally behind stable contracts so that providers, routing, task state, tools, agents, and evaluation remain independently testable.

## Current milestone

- Typed, provider-neutral chat contracts in `src/aegis_engine/models/base.py`.
- Environment-backed local configuration in `src/aegis_engine/config.py`.
- Ollama HTTP adapter in `src/aegis_engine/models/ollama.py`.
- Provider errors are normalized and marked retryable where appropriate.
- Installed model discovery reads capabilities reported by Ollama's `/api/show` endpoint; unknown capabilities remain empty.
- No model is downloaded automatically.
- Unit tests use mocked HTTP responses and do not require Ollama to be running.
- Tool definitions are centrally registered, schema-validated, allowlisted, timeout-bounded, and permission-aware.
- Initial tools are calculator, local time, word count, and approved-root text reading.
- The orchestrator can expose an explicit tool subset, execute bounded tool-call loops, and pause for confirmation.
- Optional Laya integration provides typed task understanding without replacing the generative Ollama path.

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
| `AEGIS_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama HTTP endpoint |
| `AEGIS_DEFAULT_MODEL` | `qwen2.5:7b` | Initial model candidate |
| `AEGIS_REQUEST_TIMEOUT_SECONDS` | `60` | Per-request timeout |
| `AEGIS_LOCAL_ONLY` | `true` | Keeps the first milestone local-only |
| `AEGIS_APPROVED_FILE_ROOTS` | empty | OS-separated roots allowed for text reads |
| `AEGIS_LAYA_ENABLED` | `false` | Enables optional Laya task understanding |
| `AEGIS_LAYA_MODEL` | empty | Optional Laya checkpoint name; the orchestrator uses `typed-decisions` for routing questions |
| `AEGIS_LAYA_PRELOAD` | `false` | Eagerly loads Laya checkpoints; leave disabled to load on first decision |

## Optional Laya decision engine

[Laya](https://github.com/NandhaKishorM/laya) is a typed-decision engine, not a chat model. It can classify a request's domain and difficulty and estimate whether tools or sensitive handling are needed. Aegis keeps final natural-language generation in Ollama and treats Laya's output as advisory until it has been evaluated on project-specific examples.

The dependency is optional because it brings PyTorch, Transformers, and model checkpoints. The core project does not download it or any checkpoint automatically:

```bash
uv sync --extra laya
export AEGIS_LAYA_ENABLED=true
```

The first decision may download a Laya checkpoint from Hugging Face. `AEGIS_LAYA_PRELOAD=true` makes that download happen during provider construction and is therefore not enabled by default. Do not enable it on a memory-constrained machine without checking the available RAM first.

For direct use with an injected provider:

```python
from aegis_engine.decisions import LayaDecisionEngine

decision_engine = LayaDecisionEngine()
```

Laya does not replace `qwen2.5:7b` or `deepseek-r1:8b`; those remain the local generative models used by the model gateway.

## Architecture direction

The intended dependency direction is:

```text
interfaces → orchestration → routing / agents / tools
                           → model contracts → provider adapters
                           → storage / evaluation
```

Provider-specific HTTP details stay inside provider adapters. The future orchestrator will consume `ModelProvider`, `ChatRequest`, and `ChatResponse` rather than importing Ollama directly.

## Next step

The model registry and deterministic router discover available models, preserve explicit capability metadata, enforce local-only routing, and return a selected model plus fallbacks and a routing reason. The bounded orchestration runtime records task state, selected models, results, errors, cancellation, retries, and tool iterations. No LLM-based routing or multi-agent execution is needed until this foundation is tested further.
