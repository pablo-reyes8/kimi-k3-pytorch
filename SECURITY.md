# Security policy

## Supported versions

Kimi-K3 Mini is pre-release research software. Security fixes are applied to
the latest commit on `main`; older commits and locally modified forks are not
maintained as separate supported releases.

| Version | Supported |
|---|---|
| `main` / latest `0.0.x` | Yes |
| Older snapshots | No |

## Reporting a vulnerability

Please do not open a public issue for a vulnerability. Use
[GitHub Private Vulnerability Reporting](https://github.com/pablo-reyes8/kimi-k3/security/advisories/new)
with:

- the affected commit and component;
- a minimal reproduction or proof of concept;
- expected impact and required preconditions;
- any suggested mitigation.

The maintainers will acknowledge a complete report as soon as practical,
investigate it privately and coordinate disclosure after a fix is available.

## Trust boundaries

- Treat model checkpoints as executable-adjacent input. PyTorch checkpoint
  deserialization must only be used with files from a trusted source.
- Dataset and tokenizer artifacts can contain malformed or adversarial input.
  Keep caches isolated and review external dataset provenance.
- Docker volume mounts expose host files to the container. The supplied
  inference mount is read-only by default.
- Generated text is untrusted output. This repository does not add moderation,
  sandboxing or production serving isolation.

Security reports are for vulnerabilities in this repository. Questions about
model quality, research results or installation belong in regular issues.
