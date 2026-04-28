# litellm-bot — production image.
#
# Build: docker build -t litellm-bot .
# Run:   docker run --rm -p 8000:8000 --env-file .env litellm-bot
#
# This image bundles the karpathy_check runtime: `claude` (Claude Code CLI),
# `gh` (GitHub CLI), `git`, plus a full clone of BerriAI/litellm at
# /opt/litellm — all four are required by karpathy_check.py and the default
# Render Python buildpack ships none of them, which is why prod was logging
# `karpathy_skipped reason=no_claude` on every PR.

FROM python:3.12-slim-trixie

# ---- OS deps ---------------------------------------------------------------
# - git: karpathy_check uses worktrees on the litellm clone
# - curl/ca-certificates: claude installer + GitHub CLI repo key
# - gnupg: gh apt repo signature verification
# - gh: karpathy_check shells out to `gh pr view` / `gh issue view`
# Single RUN keeps the image one layer thinner; rm -rf /var/lib/apt/lists
# at the end strips ~40MB of apt metadata we'd otherwise carry forever.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        gnupg \
 && install -dm 755 /etc/apt/keyrings \
 && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
 && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends gh \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# ---- Claude Code CLI -------------------------------------------------------
# Native installer drops the binary in /root/.local/bin. Putting that on PATH
# globally so `shutil.which("claude")` in karpathy_check.py finds it whether
# the process runs as root or under uvicorn's worker.
#
# Headless auth: the karpathy CC subprocess is invoked with
# `--permission-mode bypassPermissions`, which still requires valid credentials.
# Two options at runtime, pick one:
#   (a) Direct to Anthropic — set ANTHROPIC_API_KEY.
#   (b) Through the LiteLLM proxy (reuses the same key the rest of the bot
#       already uses) — set ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN, e.g.
#         ANTHROPIC_BASE_URL=$LITELLM_API_BASE
#         ANTHROPIC_AUTH_TOKEN=$LITELLM_API_KEY
#       Requires the proxy to expose /v1/messages (Anthropic Messages format).
#       Set ANTHROPIC_MODEL if your proxy routes a non-default model name.
# Without one of these, every karpathy run fails with `karpathy_failed err=no_json`.
RUN curl -fsSL https://claude.ai/install.sh | bash
ENV PATH="/root/.local/bin:${PATH}"

# ---- litellm clone ---------------------------------------------------------
# karpathy_check._resolve_litellm_clone() falls back to /opt/litellm when
# LITELLM_CLONE is unset, so cloning to that exact path means zero new env
# vars are required. Full clone (not --depth 1) because karpathy adds
# worktrees at arbitrary PR-head SHAs, which a shallow clone can't satisfy.
# `git fetch origin pull/N/head` runs on each karpathy invocation so a clone
# that's a few days stale at image-build time is still fine.
RUN git clone https://github.com/BerriAI/litellm.git /opt/litellm

# ---- uv + Python deps ------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies in a separate layer from the app source so that an
# app-only edit doesn't bust the dep-install cache.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_DEV=1
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

ENV PATH="/app/.venv/bin:${PATH}"

# ---- Runtime ---------------------------------------------------------------
# Render injects $PORT; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
