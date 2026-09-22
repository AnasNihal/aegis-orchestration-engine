# Initial architecture

## Boundaries

`aegis_engine.config` owns validated runtime settings. It does not perform network access or model discovery.

`aegis_engine.models.base` defines the provider-neutral contracts. These types are the seam between the orchestration engine and inference providers.

`aegis_engine.models.ollama` owns Ollama HTTP paths, payload conversion, response parsing, capability discovery, and provider error normalization. No future orchestrator module should build an Ollama request directly.

## Capability policy

The adapter reports discovered model identity, availability, local status, and capabilities returned by Ollama's model inspection endpoint. If inspection is unavailable, capabilities remain empty. Model names are never used as capability evidence. The registry can apply additional explicit configuration without overwriting provider metadata accidentally.

The deterministic router filters by availability, enabled status, local-only scope, provider, and required capabilities. It uses a configured preferred model only after those checks pass, then returns stable fallbacks. There is no LLM-based routing yet.

The orchestration runtime creates immutable task state, transitions it through planning and execution, invokes the selected provider through `ModelGateway`, and stops after bounded retry and tool-iteration counts. Tool exposure is explicit per task, and confirmation-required tools transition the task to `waiting_for_confirmation` rather than executing silently.

The tool layer is independent of the model adapter. `ToolRegistry` stores descriptions, JSON input schemas, handlers, permissions, and timeouts. `ToolExecutor` applies the task allowlist, validates arguments, requires confirmation for sensitive definitions, and returns structured results. The initial file reader requires an explicit approved root and enforces a size limit. There is no arbitrary shell tool.

## Reliability policy

Every provider call is bounded by the configured timeout. Transport failures become `ProviderError` instances, and server-side 5xx responses are marked retryable. The retry policy belongs to the future orchestrator, not to the provider adapter.

## Security policy

The initial provider accepts only a configured HTTP(S) endpoint and never logs request content, headers, or response bodies. There is no shell execution, file mutation, hosted provider, or automatic model download. File reads are disabled unless an approved root is explicitly configured.
