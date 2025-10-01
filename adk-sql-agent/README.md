# OpenTelemetry ADK instrumentation example

<!-- TODO: link to devsite doc once it is published -->

This demo shows does remote uploading and with `.ref` attributes in the telemetry, following OTel semconv 1.37 for GenAI: https://github.com/open-telemetry/semantic-conventions/blob/v1.37.0/docs/gen-ai/gen-ai-events.md

## Configure upload
This demo uses the [fsspec](https://filesystem-spec.readthedocs.io/) library to do the upload
in a vendor agnostic way, which is configured in an environment variable
`OTEL_PYTHON_GENAI_UPLOADER_PATH`. For example, to upload to GCS bucket `my-bucket` with
subdirectory `v1/`:

```console
export OTEL_PYTHON_GENAI_UPLOADER_PATH="gs://my-bucket/v1"
```

This will work with any installed fsspec implementations and also allows using any fsspec
configuration like URL Chaining:
- [Built in implementations](https://filesystem-spec.readthedocs.io/en/latest/api.html#built-in-implementations)
- [Known third party impls](https://filesystem-spec.readthedocs.io/en/latest/api.html#external-implementations)
- [URL Chaining](https://filesystem-spec.readthedocs.io/en/latest/features.html#url-chaining)

## Run the demo

To run the app, set some environment variables (alternatively can be put in `main.env`):

```console
# Port if not default
export PORT=8000
# Project for ADC
export GOOGLE_CLOUD_PROJECT=my-project

# Configure the upload path
export OTEL_PYTHON_GENAI_UPLOADER_PATH="gs://${GOOGLE_CLOUD_PROJECT}/v1"

uv run --env-file main.env main.py
```

## Dockerfile

Build

```console
# from this directory, run with repo root build context
docker build -f ./Dockerfile -t adk-sql-agent-write-aside ..
```

Run:
```console
docker run --rm \
    -p 8000:8000 \
    -e PORT=8000 \
    -e GOOGLE_APPLICATION_CREDENTIALS \
    -e GOOGLE_CLOUD_PROJECT \
    -e OTEL_PYTHON_GENAI_UPLOADER_PATH="gs://otel-starter-project-genai-refs/v1" \
    -v "$GOOGLE_APPLICATION_CREDENTIALS:$GOOGLE_APPLICATION_CREDENTIALS" \
    adk-sql-agent-write-aside:latest
```
