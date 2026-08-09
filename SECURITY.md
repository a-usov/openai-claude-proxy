# Security policy

## Supported versions

Security fixes are applied to the current default branch. Until tagged releases
exist, deployments should track a reviewed commit and a committed `uv.lock`.

## Reporting a vulnerability

Do not open a public issue for suspected credential exposure, authentication
bypass, request-routing vulnerabilities, or sensitive-data leakage. Report the
issue privately to the repository owner or your organization's security team and
include:

- The affected commit or version.
- The relevant route and configuration, with all secrets redacted.
- Reproduction steps using synthetic data.
- The expected and observed behavior.
- Your assessment of impact and whether exploitation is ongoing.

If a credential may have been exposed, revoke or rotate it immediately before
waiting for a code fix.

## Deployment guidance

- Bind to loopback unless remote access is explicitly required.
- Put shared deployments behind an authenticated, TLS-terminating gateway.
- Inject static credentials from a secrets manager, not source control, image
  layers, command histories, or `.env` files.
- Keep `VERIFY_SSL=true`; use an organizational CA bundle at the platform layer
  rather than disabling certificate verification.
- Restrict outbound networking to the configured upstream origin.
- Treat prompts, tool arguments, tool results, and model responses as sensitive
  work data. Avoid body logging and use retention-appropriate infrastructure.
- Set `PASSTHROUGH_ENABLED=false` if clients only require `/v1/messages`.
- Review `UPSTREAM_HEADERS_JSON` carefully because static headers override
  forwarded authentication headers.
- Size `MAX_REQUEST_BODY_BYTES`, `MAX_RESPONSE_BODY_BYTES`, and
  `MAX_ERROR_BODY_BYTES` for the largest expected Claude Code prompt and tool result
  while keeping bounded buffering appropriate for each worker's memory limit.
- Treat proxy error redaction as defense in depth, not permission for an upstream
  gateway to include prompts, tool results, credentials, tenant URLs, or request
  bodies in its error messages.
