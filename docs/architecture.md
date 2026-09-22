# Initial architecture

## Boundaries

`mmo_engine.config` owns validated runtime settings. It does not perform network access or model discovery.

`mmo_engine.models.base` defines the provider-neutral contracts. These types are the seam between the orchestration engine and inference providers.

`mmo_engine.models.ollama` owns Ollama HTTP paths, payload conversion, response parsing, and provider error normalization. No future orchestrator module should build an Ollama request directly.

## Capability policy

The first adapter reports discovered model identity, availability, and local status. It leaves `capabilities` empty because model names are not proof of tool-calling, structured-output, reasoning, or vision support. Capability metadata will be added by explicit registry configuration and provider checks in the next milestone.

## Reliability policy

Every provider call is bounded by the configured timeout. Transport failures become `ProviderError` instances, and server-side 5xx responses are marked retryable. The retry policy belongs to the future orchestrator, not to the provider adapter.

## Security policy

The initial provider accepts only a configured HTTP(S) endpoint and never logs request content, headers, or response bodies. There is no shell execution, file mutation, hosted provider, or automatic model download in this milestone.

