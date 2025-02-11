# https://python.langchain.com/docs/tutorials/chatbot

from os import environ
from typing import Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_vertexai import ChatVertexAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from opentelemetry import trace


def main() -> None:
    model = ChatVertexAI(
        model="gemini-1.5-flash",
        project=environ.get("GOOGLE_CLOUD_PROJECT", None),
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    class State(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        language: str

    trimmer = trim_messages(
        max_tokens=200,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )

    messages = [
        SystemMessage(content="you're a good assistant"),
        HumanMessage(content="hi! I'm bob"),
        AIMessage(content="hi!"),
        HumanMessage(content="I like vanilla ice cream"),
        AIMessage(content="nice"),
        HumanMessage(content="whats 2 + 2"),
        AIMessage(content="4"),
        HumanMessage(content="thanks"),
        AIMessage(content="no problem!"),
        HumanMessage(content="having fun?"),
        AIMessage(content="yes!"),
    ]

    workflow = StateGraph(state_schema=State)

    def call_model(state: State):
        trimmed_messages = trimmer.invoke(state["messages"])
        prompt = prompt_template.invoke(
            {"messages": trimmed_messages, "language": state["language"]}
        )
        response = model.invoke(prompt)
        return {"messages": [response]}

    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "abc567"}}
    query = "What is my name?"
    language = "English"

    input_messages = messages + [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages, "language": language},
        config,
    )
    output["messages"][-1].pretty_print()

    config = {"configurable": {"thread_id": "abc678"}}
    query = "What math problem did I ask?"
    language = "English"

    input_messages = messages + [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages, "language": language},
        config,
    )
    output["messages"][-1].pretty_print()


with trace.get_tracer(__name__).start_as_current_span("demo-root-span"):
    main()
