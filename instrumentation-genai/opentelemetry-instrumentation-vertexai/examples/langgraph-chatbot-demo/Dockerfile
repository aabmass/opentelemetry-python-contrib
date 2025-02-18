FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS build
RUN apt update && apt install -y git curl
WORKDIR /build
ARG GIT_REF
ENV SCRIPT_URL="https://raw.githubusercontent.com/aabmass/opentelemetry-python-contrib/${GIT_REF?must be set to a git commit}/instrumentation-genai/opentelemetry-instrumentation-vertexai/examples/langgraph-chatbot-demo/run_streamlit.py"
# Creates the virtualenv
RUN curl -LO "${SCRIPT_URL}" && curl -LO "${SCRIPT_URL}.lock"
RUN VIRTUAL_ENV=./venv uv sync --active --frozen --script run_streamlit.py

FROM python:3.13-slim
WORKDIR /build
COPY --from=build /build/ .
ENV PATH="/build/venv/bin:$PATH"
ENTRYPOINT ["python", "run_streamlit.py"]
