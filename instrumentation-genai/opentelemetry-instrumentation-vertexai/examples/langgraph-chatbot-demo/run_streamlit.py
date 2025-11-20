# /// script
# requires-python = ">=3.13"
# dependencies = [
#  "langgraph-chatbot-demo",
# ]
#
# [tool.uv.sources]
# langgraph-chatbot-demo = { git = "https://github.com/aabmass/opentelemetry-python-contrib.git", subdirectory = "instrumentation-genai/opentelemetry-instrumentation-vertexai/examples/langgraph-chatbot-demo", branch = "vertex-streamlit-langgraph" }
#
# ///

from langgraph_chatbot_demo.run_streamlit import run_streamlit

run_streamlit()
