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

# ---- Non-root user ---------------------------------------------------------
# claude refuses `--permission-mode bypassPermissions` under uid 0 with
# "--dangerously-skip-permissions cannot be used with root/sudo privileges".
# karpathy_check.py hard-codes bypassPermissions in _invoke_claude(), so the
# runtime MUST be non-root or every karpathy run fails with
# `karpathy_failed err=no_json rc=1`.
#
# Create the bot user before installing claude so the installer's hard-coded
# `$HOME/.local/bin` ends up under the same user that will eventually exec it.
RUN useradd --create-home --shell /bin/bash --uid 1000 bot

# ---- litellm clone ---------------------------------------------------------
# karpathy_check._resolve_litellm_clone() falls back to /opt/litellm when
# LITELLM_CLONE is unset, so cloning to that exact path means zero new env
# vars are required. Full clone (not --depth 1) because karpathy adds
# worktrees at arbitrary PR-head SHAs, which a shallow clone can't satisfy.
# `git fetch origin pull/N/head` runs on each karpathy invocation so a clone
# that's a few days stale at image-build time is still fine. Cloned as bot
# so the per-call `git fetch` and `git worktree add` succeed without sudo.
RUN install -d -o bot -g bot /opt/litellm \
 && su bot -c "git clone https://github.com/BerriAI/litellm.git /opt/litellm"

# ---- Claude Code CLI -------------------------------------------------------
# Installed under bot's $HOME (not root's) so the binary lives at
# /home/bot/.local/bin/claude — the path the runtime user can actually exec.
# Headless auth options (set one as a Render env var):
#   (a) ANTHROPIC_API_KEY (direct to Anthropic).
#   (b) ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN (LiteLLM proxy — reuses
#       the existing LITELLM_API_KEY). Set ANTHROPIC_MODEL too if your
#       proxy doesn't route the default claude-sonnet-* model name.
USER bot
ENV HOME=/home/bot
RUN curl -fsSL https://claude.ai/install.sh | bash
ENV PATH="/home/bot/.local/bin:${PATH}"
USER root

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

# ---- skills submodule ------------------------------------------------------
# `skills/pr-review-agent-skills` is a git submodule in the parent repo, so
# `COPY . /app` pulls in only the empty placeholder directory (.git is in
# .dockerignore, so `git submodule update` from inside the container can't
# work either). Clone the submodule directly and pin to the SHA the parent
# repo gitlinks, so the skill never drifts out from under the bot.
# karpathy_check._resolve_skill_path() looks for SKILL.md at this exact path
# under DEFAULT_SKILL — without it, every karpathy run logs
# `karpathy_skipped reason=no_skill`.
# Pinned to a SHA that contains litellm-karpathy-check/SKILL.md. The parent
# repo's gitlink (17797eb) predates the karpathy skill being added upstream,
# so we deliberately move past the gitlinked commit here. Bump when the
# upstream skill content changes.
ARG SKILLS_SHA=7ad6f131d90b151ff98373b6583f51c483cb2826
RUN rm -rf /app/skills/pr-review-agent-skills \
 && git clone https://github.com/BerriAI/pr-review-agent-skills.git /app/skills/pr-review-agent-skills \
 && git -C /app/skills/pr-review-agent-skills checkout --detach "${SKILLS_SHA}"

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

ENV PATH="/app/.venv/bin:${PATH}"

# ---- Runtime ---------------------------------------------------------------
# Drop to the bot user for CMD — see the claude-no-root note above the
# Claude Code install. /app stays root-owned (bot only needs read; karpathy's
# tempdirs go to /tmp) and /opt/litellm is bot-owned from the clone step so
# `git fetch origin pull/N/head` works without sudo.
USER bot

# Render injects $PORT; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
