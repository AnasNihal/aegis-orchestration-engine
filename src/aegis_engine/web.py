"""Small local web interface backed by the Aegis orchestrator.

This intentionally uses the Python standard library. It is a local
development interface, not yet a production web server.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from aegis_engine.config import Settings
from aegis_engine.models import (
    ChatMessage,
    DeterministicModelRouter,
    ModelGateway,
    ModelInfo,
    ModelRegistry,
    OllamaProvider,
)
from aegis_engine.orchestration import Orchestrator
from aegis_engine.tasks import TaskStatus


MAX_BODY_BYTES = 1_000_000
MAX_HISTORY_MESSAGES = 40
MAX_MESSAGE_CHARS = 32_000


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aegis Local Assistant</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #111827; color: #e5e7eb; }
    main { max-width: 900px; margin: 0 auto; min-height: 100vh; display: flex; flex-direction: column; }
    header { padding: 24px 20px 12px; border-bottom: 1px solid #374151; }
    h1 { margin: 0 0 8px; font-size: 1.5rem; }
    .subtitle { color: #9ca3af; }
    #messages { flex: 1; padding: 20px; overflow-y: auto; }
    .message { white-space: pre-wrap; padding: 12px 14px; border-radius: 10px; margin: 10px 0; line-height: 1.5; }
    .user { background: #1d4ed8; margin-left: 15%; }
    .assistant { background: #1f2937; margin-right: 15%; }
    .controls { padding: 16px 20px 24px; border-top: 1px solid #374151; }
    select, textarea, button { font: inherit; border-radius: 8px; border: 1px solid #4b5563; }
    select, textarea { background: #111827; color: #e5e7eb; padding: 10px; }
    select { width: 100%; margin-bottom: 10px; }
    textarea { width: 100%; min-height: 80px; box-sizing: border-box; resize: vertical; }
    button { background: #2563eb; color: white; padding: 10px 18px; margin-top: 10px; cursor: pointer; }
    button:disabled { opacity: .5; cursor: wait; }
    #status { color: #9ca3af; font-size: .9rem; margin-top: 8px; }
  </style>
</head>
<body>
<main>
  <header><h1>Aegis Local Assistant</h1><div class="subtitle">Local Ollama orchestration engine</div></header>
  <section id="messages"><div class="message assistant">Choose a model and ask a question.</div></section>
  <section class="controls">
    <select id="model" aria-label="Model"></select>
    <textarea id="prompt" placeholder="Write your prompt here..."></textarea>
    <button id="send">Send</button>
    <div id="status"></div>
  </section>
</main>
<script>
const model = document.querySelector('#model');
const prompt = document.querySelector('#prompt');
const send = document.querySelector('#send');
const messages = document.querySelector('#messages');
const status = document.querySelector('#status');
const history = [];

function addMessage(role, content) {
  const node = document.createElement('div');
  node.className = 'message ' + role;
  node.textContent = content;
  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
}

async function loadModels() {
  const response = await fetch('/api/models');
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Could not load models');
  model.replaceChildren(...data.models.map(item => {
    const option = document.createElement('option');
    option.value = item.model_id;
    option.textContent = item.model_id + (item.capabilities.length ? ' (' + item.capabilities.join(', ') + ')' : '');
    return option;
  }));
}

async function sendMessage() {
  const message = prompt.value.trim();
  if (!message) return;
  const priorHistory = history.slice();
  addMessage('user', message);
  history.push({role: 'user', content: message});
  prompt.value = '';
  send.disabled = true;
  status.textContent = 'Aegis is thinking...';
  let assistantNode = null;
  let assistantText = '';
  try {
    const response = await fetch('/api/chat', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({model: model.value, message, history: priorHistory})});
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error || 'Request failed');
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const chunk = await reader.read();
      buffer += decoder.decode(chunk.value || new Uint8Array(), {stream: !chunk.done});
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === 'token') {
          if (!assistantNode) {
            assistantNode = document.createElement('div');
            assistantNode.className = 'message assistant';
            messages.appendChild(assistantNode);
          }
          assistantText += event.content;
          assistantNode.textContent = assistantText;
          messages.scrollTop = messages.scrollHeight;
        } else if (event.type === 'complete') {
          history.push({role: 'assistant', content: assistantText});
          status.textContent = 'Completed with ' + event.models.join(', ');
        } else if (event.type === 'error') {
          throw new Error(event.error || 'Request failed');
        }
      }
      if (chunk.done) break;
    }
  } catch (error) {
    addMessage('assistant', 'Error: ' + error.message);
    status.textContent = 'Request failed';
  } finally {
    send.disabled = false;
    prompt.focus();
  }
}

send.addEventListener('click', sendMessage);
prompt.addEventListener('keydown', event => { if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) sendMessage(); });
loadModels().catch(error => { status.textContent = error.message; });
</script>
</body>
</html>
"""


def parse_history(value: Any) -> tuple[ChatMessage, ...]:
    """Validate browser-provided conversation history."""

    if value is None:
        return ()
    if not isinstance(value, list) or len(value) > MAX_HISTORY_MESSAGES:
        raise ValueError("history must be a list of at most 40 messages")
    messages: list[ChatMessage] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("history messages must be objects")
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise ValueError("history supports only user and assistant text messages")
        if len(content) > MAX_MESSAGE_CHARS:
            raise ValueError("history message is too long")
        messages.append(ChatMessage(role=role, content=content))
    return tuple(messages)


def create_server(config: Settings, *, host: str = "127.0.0.1", port: int = 8765):
    """Create the local HTTP server after discovering Ollama models."""

    provider = OllamaProvider(config)
    registry = ModelRegistry()
    registry.refresh(provider)
    gateway = ModelGateway([provider])
    router = DeterministicModelRouter(registry, preferred_model=config.default_model)
    orchestrator = Orchestrator(gateway, router, provider_settings=config)
    models = tuple(model for model in registry.available() if "completion" in model.capabilities)

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Mapping[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self) -> None:
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            path = urlparse(self.path).path
            if path == "/":
                self._send_html()
            elif path == "/api/health":
                self._send_json({"status": "ok"})
            elif path == "/api/models":
                self._send_json({"models": [model_payload(model) for model in models]})
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if urlparse(self.path).path != "/api/chat":
                self._send_json({"error": "not found"}, 404)
                return
            stream_started = False
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_BODY_BYTES:
                    raise ValueError("request body is empty or too large")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, Mapping):
                    raise ValueError("request body must be a JSON object")
                message = payload.get("message")
                if not isinstance(message, str) or not message.strip():
                    raise ValueError("message is required")
                if len(message) > MAX_MESSAGE_CHARS:
                    raise ValueError("message is too long")
                model_id = payload.get("model")
                if not isinstance(model_id, str) or not model_id.strip():
                    raise ValueError("model is required")
                history = parse_history(payload.get("history"))
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                stream_started = True

                def emit(event: Mapping[str, Any]) -> None:
                    self.wfile.write(json.dumps(event, default=str).encode("utf-8") + b"\n")
                    self.wfile.flush()

                task = orchestrator.run(
                    message,
                    model_id=model_id,
                    conversation=history,
                    on_token=lambda token: emit({"type": "token", "content": token}),
                )
                if task.status is not TaskStatus.COMPLETED:
                    emit({"type": "error", "error": task.errors[-1] if task.errors else str(task.status)})
                    return
                emit(
                    {
                        "type": "complete",
                        "task_id": task.task_id,
                        "status": task.status,
                        "models": list(task.selected_models),
                    }
                )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                if stream_started:
                    self.wfile.write(json.dumps({"type": "error", "error": str(exc)}).encode("utf-8") + b"\n")
                    self.wfile.flush()
                else:
                    self._send_json({"error": str(exc)}, 400)
            except Exception as exc:
                if stream_started:
                    self.wfile.write(json.dumps({"type": "error", "error": str(exc)}).encode("utf-8") + b"\n")
                    self.wfile.flush()
                else:
                    self._send_json({"error": str(exc)}, 500)

        def log_message(self, format: str, *args: object) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


def model_payload(model: ModelInfo) -> dict[str, Any]:
    return {
        "model_id": model.model_id,
        "provider": model.provider,
        "capabilities": sorted(model.capabilities),
        "local": model.local,
    }


def serve(config: Settings, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = create_server(config, host=host, port=port)
    print(f"Aegis web interface: http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
