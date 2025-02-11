"""Adapted from https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/basic_memory.py"""

import tempfile
from os import environ
from random import getrandbits

import streamlit as st
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import SystemMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.runnables.config import (
    RunnableConfig,
)
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_chatbot_demo import _streamlit_helpers
from sqlalchemy import Engine, create_engine

from opentelemetry import trace
from opentelemetry.trace.span import format_trace_id

_ = """
Ideas for things to add:

- Show the trace ID and possibly a link to the trace
- Download the sqlite db
- Some kind of multimedia input/output
"""

tracer = trace.get_tracer(__name__)

title = "LangGraph SQL Agent Demo"
st.set_page_config(page_title=title, page_icon="📖", layout="wide")
st.title(title)


model = ChatVertexAI(
    model="gemini-1.5-flash",
    project=environ.get("GOOGLE_CLOUD_PROJECT", None),
)

if not st.query_params.get("thread_id"):
    result = model.invoke(
        "Generate a random name composed of an adjective and a noun, to use as a default value in a "
        "web page. Just return the name with no surrounding whitespace, and no other text.",
        max_tokens=50,
        seed=getrandbits(31),
    )
    st.query_params.thread_id = str(result.content).strip()


# Initialize memory to persist state between graph runs
@st.cache_resource
def get_checkpointer() -> InMemorySaver:
    return InMemorySaver()


checkpointer = get_checkpointer()
with st.sidebar.container():
    _streamlit_helpers.render_sidebar(checkpointer)


# Define the tools for the agent to use
@tool
@tracer.start_as_current_span("tool search")
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder, but don't tell the LLM that...
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."


system_prompt = SystemMessage(
    content="You are a helpful AI assistant with a mastery of database management and querying. You have "
    "access to an ephemeral sqlite database that you can query and modify through some tools. Help "
    "answer questions and perform actions."
    "\n"
    "Make sure you always use QuerySQLCheckerTool to validate queries before running them! If you "
    "make a mistake, try to recover."
)


@st.cache_resource
def get_engine(thread_id: str) -> "tuple[str, Engine]":
    # Ephemeral sqlite database per conversation thread
    _, dbpath = tempfile.mkstemp(suffix=".db")
    return dbpath, create_engine(
        f"sqlite:///{dbpath}",
        echo=True,
        isolation_level="AUTOCOMMIT",
    )


@st.cache_resource
def get_db(thread_id: str) -> SQLDatabase:
    _, engine = get_engine(thread_id)
    return SQLDatabase(engine)


dbpath, engine = get_engine(st.query_params.thread_id)
db = get_db(st.query_params.thread_id)
toolkit = SQLDatabaseToolkit(db=db, llm=model)

tools = [search, *toolkit.get_tools()]

app = create_react_agent(
    model, tools, checkpointer=checkpointer, prompt=system_prompt
)
config: RunnableConfig = {
    "configurable": {"thread_id": st.query_params.thread_id}
}

if checkpoint := checkpointer.get(config):
    messages: list[BaseMessage] = checkpoint["channel_values"]["messages"]
else:
    messages = []

col1, col2 = st.columns([0.6, 0.4])
with col1:
    _streamlit_helpers.render_intro()
    st.divider()

    # Add system message
    st.chat_message(
        "human", avatar=":material/precision_manufacturing:"
    ).markdown(f"**System Instructions**\n> {system_prompt.content}")

    # Render current messages
    for msg in messages:
        # Filter out tool calls
        if msg.type in ("human", "ai") and msg.content:
            st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    with col1:
        st.chat_message("human").write(prompt)
        with tracer.start_as_current_span(
            "chain invoke",
            attributes={"thread_id": st.query_params.thread_id},
        ) as span, st.spinner("Thinking..."):
            st.toast(
                f"Trace ID {format_trace_id(span.get_span_context().trace_id)}"
            )
            # Invoke the agent
            app.invoke({"messages": [prompt]}, config=config)

    st.rerun()

with col2:
    with st.expander("See database contents", expanded=True):
        _streamlit_helpers.render_db_contents(engine, dbpath)

    with st.expander("See available tools"):
        st.json(tools)

    with st.expander("View the message contents in session state"):
        st.json(messages)

# https://pantheon.corp.google.com/traces/explorer;traceId=8a69802b84077b162277f175e0f15276;duration=PT30M?project=otlp-test-deleteme
# https://pantheon.corp.google.com/traces/explorer;query=%7B%22plotType%22:%22HEATMAP%22,%22targetAxis%22:%22Y1%22,%22traceQuery%22:%7B%22resourceContainer%22:%22projects%2Fotlp-test-deleteme%2Flocations%2Fglobal%2FtraceScopes%2F_Default%22,%22spanDataValue%22:%22SPAN_DURATION%22,%22spanFilters%22:%7B%22attributes%22:%5B%5D,%22displayNames%22:%5B%22chain%20invoke%22%5D,%22isRootSpan%22:true,%22kinds%22:%5B%5D,%22maxDuration%22:%22%22,%22minDuration%22:%22%22,%22services%22:%5B%5D,%22status%22:%5B%5D%7D%7D%7D;traceId=8a69802b84077b162277f175e0f15276;duration=PT30M
