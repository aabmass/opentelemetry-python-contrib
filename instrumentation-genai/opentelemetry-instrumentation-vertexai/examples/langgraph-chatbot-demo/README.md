This sample contains part of the LangGraph chatbot demo taken from
https://python.langchain.com/docs/tutorials/chatbot, running with OTel instrumentation. It
sends traces and logs to the OTel collector which sends them to GCP. Docker compose wraps
everything to make it easy to run.

## Running the example

I recommend running in Cloud Shell, it's super simple. You will see GenAI spans in trace
explorer right away. Make sure the Vertex and Trace APIs are enabled in the project.

### Cloud Shell or GCE

```sh
git clone --branch=vertex-langgraph https://github.com/aabmass/opentelemetry-python-contrib.git
cd opentelemetry-python-contrib/instrumentation-genai/opentelemetry-instrumentation-vertexai/examples/langgraph-chatbot-demo
docker compose up --build --abort-on-container-exit
```

### Locally with Application Default Credentials

```sh
git clone --branch=vertex-langgraph https://github.com/aabmass/opentelemetry-python-contrib.git
cd opentelemetry-python-contrib/instrumentation-genai/opentelemetry-instrumentation-vertexai/examples/langgraph-chatbot-demo

# Export the credentials to `GOOGLE_APPLICATION_CREDENTIALS` environment variable so it is
# available inside the docker containers
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/application_default_credentials.json
# Lets collector read mounted config
export USERID="$(id -u)"
# Specify the project ID
export GOOGLE_CLOUD_PROJECT=<your project id>
docker compose up --build --abort-on-container-exit
```
