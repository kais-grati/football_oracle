# Paper trader image for a small VPS.
ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Madrid

RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG UID=1000
ARG GID=1000
RUN groupadd --gid "${GID}" app \
    && useradd --uid "${UID}" --gid "${GID}" \
       --create-home --shell /bin/false app

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY paper_trader.py simulator.py ./
COPY papertrader ./papertrader
COPY football_matches_odds.csv match_features.csv ./

RUN mkdir -p /app/data/paper && chown -R app:app /app

USER app

# The scheduler runs immediately; PAPER_TICK_SECONDS controls the loop (30s).
CMD ["python", "paper_trader.py", "run"]
