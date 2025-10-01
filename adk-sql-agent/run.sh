#!/bin/sh
uv run --env-file main.env adk web --otel_to_cloud --host 0.0.0.0
