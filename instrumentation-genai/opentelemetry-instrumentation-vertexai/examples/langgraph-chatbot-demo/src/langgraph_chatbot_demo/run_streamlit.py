import importlib.util
import os
import subprocess

import google.auth
import google.auth.transport
import google.auth.transport.requests


def setenv_default(k: str, v: str) -> None:
    if k not in os.environ:
        os.environ[k] = v


def run_streamlit() -> None:
    creds, project_id = google.auth.default()
    creds.refresh(google.auth.transport.requests.Request())

    setenv_default(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "https://telemetry.googleapis.com:443"
    )
    setenv_default("OTEL_SERVICE_NAME", "langgraph-chatbot-demo")
    setenv_default("OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED", "true")
    setenv_default(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
    )
    setenv_default("OTEL_LOGS_EXPORTER", "gcp_logging")
    setenv_default("OTEL_TRACES_EXPORTER", "otlp_google_auth")
    setenv_default("OTEL_RESOURCE_ATTRIBUTES", f"gcp.project_id={project_id}")

    # Hide metadata and token refreshes
    setenv_default(
        "OTEL_PYTHON_EXCLUDED_URLS",
        "computeMetadata,oauth2.googleapis.com",
    )
    # subprocess.run(["opentelemetry-instrument", "ipython"])

    langchain_app_spec = importlib.util.find_spec(
        "langgraph_chatbot_demo.langchain_history"
    )
    if not (langchain_app_spec and langchain_app_spec.origin):
        raise Exception("Could not find langchain_history.py")

    print(f"Starting langchain app {langchain_app_spec.origin}")

    subprocess.run(
        [
            "opentelemetry-instrument",
            "streamlit",
            "run",
            "--client.toolbarMode=developer",
            langchain_app_spec.origin,
        ],
        check=True,
    )
