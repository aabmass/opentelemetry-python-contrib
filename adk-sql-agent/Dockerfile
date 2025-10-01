# NOTE: this must be run from the repo root as build context

# Adapted from https://docs.astral.sh/uv/guides/integration/docker/#intermediate-layers
# Install uv
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y git

# Change the working directory to the `app` directory
WORKDIR /app

# Copy the project into the image
COPY . ./

WORKDIR /app/adk-sql-agent

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked

ENTRYPOINT [ "uv", "run", "--locked", "--env-file=main.env", "main.py" ]
