# SDK conformance checks

These checks validate proxy wire objects with the pinned provider SDK releases.
Both SDKs are development dependencies, so they are installed by the normal locked
development sync and remain outside the proxy's runtime dependencies.

Reviewed revisions:

- `anthropic-sdk-python` 0.121.0, commit
  `009b035305e0724ce108ebd796935f91711fc6e1`
- `openai-python` 2.53.0, commit
  `0c09a3fe815184f0a46fbf18b1aba84a467c854e`

The checks are part of the normal pytest collection:

```bash
uv sync --locked
uv run pytest -q
```

The tests deliberately fail on an SDK version mismatch so schema drift requires an
explicit review. The Anthropic tests validate complete Messages, usage, content
blocks, token-count results, error envelopes, model pages, and every emitted SSE
object. The OpenAI tests validate complete Chat/Responses response and stream event
models—including Responses sequence numbers—and run `AsyncOpenAI` through a
loopback Uvicorn instance for buffered and streamed Responses, Chat Completions,
model listing, embedding pass-through, and error metadata. The real-HTTP test skips
only in restricted environments that prohibit listening sockets.

The release versions are locked in `uv.lock`; the reviewed source commit hashes above
record the exact source trees used during implementation review. Updating either SDK
requires reviewing schema drift, updating the pinned development dependency and
hash together, and rerunning pytest.
