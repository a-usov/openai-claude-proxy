# Representative multi-turn wire fixtures

These sanitized examples show the important request shapes at the proxy/upstream
boundary. IDs, prompts, tool results, models, and credentials are synthetic. The
executable equivalents live in `tests/integration/` and the full streamed provider
events live in `tests/fixtures/`.

## Chat Completions tool turn

Claude sends the first turn to `POST /v1/messages`:

```json
{
  "model": "claude-work",
  "max_tokens": 128,
  "stream": true,
  "messages": [{"role": "user", "content": "Read the synthetic file"}],
  "tools": [{
    "name": "read_file",
    "input_schema": {
      "type": "object",
      "properties": {"path": {"type": "string"}},
      "required": ["path"]
    }
  }]
}
```

After model mapping, the upstream Chat request is:

```json
{
  "model": "gpt-work",
  "messages": [{"role": "user", "content": "Read the synthetic file"}],
  "max_tokens": 128,
  "stream": true,
  "tools": [{
    "type": "function",
    "function": {
      "name": "read_file",
      "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"]
      }
    }
  }],
  "stream_options": {"include_usage": true}
}
```

Claude returns the tool result in its transcript. The second upstream request
replays the assistant call and appends the corresponding tool message:

```json
{
  "model": "gpt-work",
  "messages": [
    {"role": "user", "content": "Read the synthetic file"},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [{
        "id": "call_read",
        "type": "function",
        "function": {"name": "read_file", "arguments": "{\"path\":\"synthetic.txt\"}"}
      }]
    },
    {"role": "tool", "tool_call_id": "call_read", "content": "synthetic contents"}
  ],
  "max_tokens": 128,
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

See `tests/fixtures/chat/tool_call.sse` and
`tests/fixtures/chat/final_answer.sse` for the upstream stream fixtures.

## Responses stateless reasoning and tool turn

The equivalent initial Responses request uses typed input items and disables
provider-side retention by default:

```json
{
  "model": "gpt-work",
  "input": [{
    "type": "message",
    "role": "user",
    "content": [{"type": "input_text", "text": "Read the synthetic file"}]
  }],
  "max_output_tokens": 128,
  "stream": true,
  "store": false,
  "reasoning": {"effort": "high"},
  "tools": [{
    "type": "function",
    "name": "read_file",
    "parameters": {
      "type": "object",
      "properties": {"path": {"type": "string"}},
      "required": ["path"]
    }
  }]
}
```

Suppose `response.output` contains an encrypted reasoning item followed by a
function call. The proxy returns visible `tool_use` content plus a proxy-marked,
base64 state carrier. On the next turn it validates the carrier against the visible
content and the currently mapped upstream model, then reconstructs this request:

```json
{
  "model": "gpt-work",
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": [{"type": "input_text", "text": "Read the synthetic file"}]
    },
    {
      "type": "reasoning",
      "id": "rs_synthetic",
      "summary": [],
      "encrypted_content": "opaque-synthetic-state"
    },
    {
      "type": "function_call",
      "id": "fc_read",
      "call_id": "call_read",
      "name": "read_file",
      "arguments": "{\"path\":\"synthetic.txt\"}"
    },
    {
      "type": "function_call_output",
      "call_id": "call_read",
      "output": "synthetic contents"
    }
  ],
  "max_output_tokens": 128,
  "stream": true,
  "store": false,
  "reasoning": {"effort": "high"}
}
```

The carrier itself is intentionally not copied into documentation because it can
contain visible model output and tool arguments. Executable carrier generation and
validation are covered by `tests/test_reasoning_state.py`; complete SSE fixtures are
under `tests/fixtures/responses/`.

## Anthropic pass-through

With `UPSTREAM_PROTOCOL=anthropic`, the proxy does not translate the body. For
example, these exact bytes are sent to the configured `/messages` path:

```json
{"model":"claude-upstream","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"Synthetic turn one"},{"role":"assistant","content":"Synthetic answer one"},{"role":"user","content":"Synthetic turn two"}]}
```

`Anthropic-Version`, `Anthropic-Beta`, helper authentication, client query
parameters, configured query parameters, response status, retry headers, JSON error
bodies, and SSE bytes are preserved according to their explicit allowlists. The
byte-for-byte multi-turn fixture is executable in
`tests/integration/test_anthropic_passthrough.py`.
