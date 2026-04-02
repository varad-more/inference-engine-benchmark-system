# Security Policy

## Scope

This project is a benchmarking tool that runs local Docker containers and makes HTTP requests to local inference servers. There is no authentication, no user data, and no network-facing production service.

## Reporting a Vulnerability

If you find a security issue (e.g. command injection in the runner, insecure Docker configuration, credential exposure), please open a GitHub Issue marked **[Security]** or contact the maintainer directly.

## Secrets Handling

- HuggingFace tokens are read from `.env` and passed as environment variables to Docker containers
- `.env` is listed in `.gitignore` — never commit it
- No tokens are written to result JSON files or logs
