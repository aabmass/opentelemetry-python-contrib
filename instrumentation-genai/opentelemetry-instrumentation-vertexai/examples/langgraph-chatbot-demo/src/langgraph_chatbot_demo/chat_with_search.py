"""Adapated from https://github.com/streamlit/llm-examples/blob/main/pages/2_Chat_with_search.py"""

from os import environ

import streamlit as st
from langchain.agents import AgentType, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_vertexai import ChatVertexAI

st.title("🔎 LangChain - Chat with search")

"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I'm a chatbot who can search the web. How can I help you?",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(
    placeholder="Who won the Women's U.S. Open in 2018?"
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatVertexAI(
        model="gemini-1.5-flash",
        project=environ.get("GOOGLE_CLOUD_PROJECT", None),
    )
    search = DuckDuckGoSearchRun(name="Search")
    search_agent = initialize_agent(
        [search],
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
    )
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(
            st.container(), expand_new_thoughts=False
        )
        response = search_agent.run(
            st.session_state.messages, callbacks=[st_cb]
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
        st.write(response)
