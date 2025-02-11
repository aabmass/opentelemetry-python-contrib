This sample contains a Streamlit + LangGraph chatbot demo. It sends traces and logs to the GCP
with the OTLP exporter and opentelemetry-exporter-gcp-logging exporters.

The `run_streamlit.py` script allows you to easily run the sample with auto instrumentation
enabled and sending telemetry to GCP. It just sets some environment variables and runs with
`opentelemetry-instrument.

## Running the example

First, make sure you have `uv` installed: https://docs.astral.sh/uv/getting-started/installation/.

Optionally, set a project with `export GOOGLE_CLOUD_PROJECT=...`. The app respects ADC.

### Without cloning

```sh
uv run --upgrade https://raw.githubusercontent.com/aabmass/opentelemetry-python-contrib/refs/heads/vertex-langgraph/instrumentation-genai/opentelemetry-instrumentation-vertexai/examples/langgraph-chatbot-demo/run_streamlit.py
```

### With cloned repo

```sh
git clone --branch=vertex-langgraph https://github.com/aabmass/opentelemetry-python-contrib.git
cd opentelemetry-python-contrib/instrumentation-genai/opentelemetry-instrumentation-vertexai/examples/langgraph-chatbot-demo
uv run run_streamlit_local.py
```

### Without auto instrumentation

```sh
uv run streamlit run src/langgraph_chatbot_demo/langchain_history.py
```
