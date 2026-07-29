# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS builder
WORKDIR /build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md LICENSE ./
COPY configuration ./configuration
COPY data ./data
COPY inference ./inference
COPY scripts ./scripts
COPY src ./src
COPY training ./training

RUN python -m pip install --upgrade pip \
    && python -m pip wheel --wheel-dir /wheels ".[data]"


FROM python:${PYTHON_VERSION}-slim AS runtime
ARG APP_UID=10001
ARG APP_GID=10001
LABEL org.opencontainers.image.title="Kimi-K3 Mini" \
      org.opencontainers.image.description="Research-scale Kimi K3 architecture, training and native-cache inference" \
      org.opencontainers.image.source="https://github.com/pablo-reyes8/kimi-k3" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --gid "${APP_GID}" kimi \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" \
        --create-home --shell /usr/sbin/nologin kimi

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels \
        "kimi-k3-mini[data]" \
    && rm -rf /wheels

WORKDIR /app
COPY --chown=kimi:kimi config ./config
COPY --chown=kimi:kimi scripts ./scripts
COPY --chown=kimi:kimi LICENSE README.md ./
RUN mkdir -p checkpoints outputs data/cache \
    && chown -R kimi:kimi checkpoints outputs data

USER kimi
CMD ["python", "-m", "scripts.train_kimi", "--profile", "config/kimi_full_pipeline/cpu_smoke", "--validate-only"]


FROM runtime AS test
USER root
RUN python -m pip install "pytest>=8"
COPY --chown=kimi:kimi tests ./tests
USER kimi
CMD ["python", "-m", "pytest", "tests/configuration", "tests/inference"]
