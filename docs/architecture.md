# Initial architecture

## Boundaries

`aegis_engine.config` owns validated runtime settings. It does not perform network access or model discovery.

`aegis_engine.models.base` defines the provider-neutral contracts. These types are the seam between the orchestration engine and inference providers.

`aegis_engine.models.ollama` owns Ollama HTTP paths, payload conversion, response parsing, capability discovery, and provider error normalization. No future orchestrator module should build an Ollama request directly.

`aegis_engine.decisions` defines a separate contract for typed decision engines. This is intentionally different from `ModelProvider`: a decision engine returns structured classifications, while a chat model returns natural-language content and may request tools. The optional Laya adapter is lazy and disabled by default.

## Capability policy

The adapter reports discovered model identity, availability, local status, and capabilities returned by Ollama's model inspection endpoint. If inspection is unavailable, capabilities remain empty. Model names are never used as capability evidence. The registry can apply additional explicit configuration without overwriting provider metadata accidentally.

The deterministic router filters by availability, enabled status, local-only scope, provider, and required capabilities. It uses a configured preferred model only after those checks pass, then returns stable fallbacks. Laya currently provides advisory request-understanding signals before generation; it does not override deterministic routing or permissions.

The orchestration runtime creates immutable task state, transitions it through planning and execution, invokes the selected provider through `ModelGateway`, and stops after bounded retry and tool-iteration counts. Tool exposure is explicit per task, and confirmation-required tools transition the task to `waiting_for_confirmation` rather than executing silently.

When enabled, the runtime records Laya's typed request-understanding result as an `understand_request` task result. A Laya failure is recorded as an advisory error and does not disable the Ollama generation path.

The tool layer is independent of the model adapter. `ToolRegistry` stores descriptions, JSON input schemas, handlers, permissions, and timeouts. `ToolExecutor` applies the task allowlist, validates arguments, requires confirmation for sensitive definitions, and returns structured results. The initial file reader requires an explicit approved root and enforces a size limit. There is no arbitrary shell tool.

## Persistence

`storage.sqlite` implements the task state store with a small versioned schema and JSON columns for immutable collections. The CLI and browser interface use the configured SQLite path; tests can continue to inject the in-memory store. Task state is deliberately separate from conversation context, long-term memory, and evaluation records.

`memory.sqlite` implements explicit note storage with title/content search and deletion. The orchestrator does not automatically save every conversation, and semantic/vector retrieval is intentionally not included until it provides a measurable benefit.

## API boundary

`api.py` is the HTTP adapter built with FastAPI. Pydantic models validate incoming chat payloads and constrain history size/content. `/api/health` and `/api/models` are typed read endpoints; `/api/chat` runs the existing orchestrator in a worker thread and streams NDJSON events without moving model or tool policy into the web layer. FastAPI's OpenAPI output is available at `/docs`.

Tool selection is currently deterministic and conservative. The request selector chooses a small safe subset from obvious phrases; the orchestrator still validates schemas, checks the task allowlist, and enforces permissions before execution. It does not grant shell access or arbitrary filesystem access.

## Agent boundary

`agents.runtime` exposes explicit coding, analysis, and research profiles over the shared model gateway. An agent receives only its assigned task and context, has a declared capability boundary and input limit, performs one bounded inference, and returns an `AgentResult`. Agents cannot spawn other agents or execute tools in this milestone.

## Verification and evaluation

`verification.checks` validates completion from recorded application evidence, not from model claims. `verification.errors` maps provider and execution failures into stable categories. `evaluation.benchmark` runs repeatable cases against selected models and records latency, success, output score, tool-call validity, and token usage when the provider reports it.

## Reliability policy

Every provider call is bounded by the configured timeout. Transport failures become `ProviderError` instances, and server-side 5xx responses are marked retryable. The retry policy belongs to the future orchestrator, not to the provider adapter.

## Security policy

The initial provider accepts only a configured HTTP(S) endpoint and never logs request content, headers, or response bodies. There is no shell execution, file mutation, hosted provider, or automatic model download. File reads are disabled unless an approved root is explicitly configured. Enabling Laya is opt-in because its package and Hugging Face checkpoints add substantial local dependencies.
