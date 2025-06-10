# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.adk.agents import Agent
import tempfile
from .tools import create_run_sql_tool


import sqlite3

from opentelemetry import trace

# from utils import ask_prompt, console, print_markdown, render_messages


SYSTEM_PROMPT = f"""\
You are a helpful AI assistant with a mastery of database design and querying. You have access
to an ephemeral sqlite3 database that you can query and modify through some tools. Help answer
questions and perform actions. Follow these rules:

- Make sure you always use sql_db_query_checker to validate SQL statements **before** running
  them. In pseudocode: `checked_query = sql_db_query_checker(query);
  sql_db_query(checked_query)`.
- Be creative and don't ask for permission! The database is ephemeral so it's OK to make some mistakes.
- The sqlite version is {sqlite3.sqlite_version} which supports multiple row inserts.
- Always prefer to insert multiple rows in a single call to the sql_db_query tool, if possible.
- You may request to execute multiple sql_db_query tool calls which will be run in parallel.

If you make a mistake, try to recover."""

INTRO_TEXT = """\
Starting agent using ephemeral SQLite DB {dbpath}. This demo allows you to chat with an Agent
that has full access to an ephemeral SQLite database. The database is initially empty. It is
built with the the LangGraph prebuilt **ReAct Agent** and the **SQLDatabaseToolkit**. Here are some samples you can try:

**Weather**
- Create a new table to hold weather data.
- Populate the weather database with 20 example rows.
- Add a new column for weather observer notes

**Pets**
- Create a database table for pets including an `owner_id` column.
- Add 20 example rows please.
- Create an owner table.
- Link the two tables together, adding new columns, values, and rows as needed.
- Write a query to join these tables and give the result of owners and their pets.
- Show me the query, then the output as a table

---
"""

tracer = trace.get_tracer(__name__)


def get_dbpath(thread_id: str) -> str:
    # Ephemeral sqlite database per conversation thread
    _, path = tempfile.mkstemp(suffix=".db")
    return path


# TODO: how to get the session from within a callback.
dbpath = get_dbpath("default")

root_agent = Agent(
    name="weather_time_agent",
    model="gemini-2.0-flash",
    description=INTRO_TEXT,
    instruction=SYSTEM_PROMPT,
    tools=[create_run_sql_tool(dbpath)],
)


# def run_agent(*, model_name: str, recursion_limit: int = 50) -> None:
#     model = PatchedChatVertexAI(model=model_name)
#     checkpointer = InMemorySaver()

#     # Ephemeral sqlite database per run
#     _, dbpath = tempfile.mkstemp(suffix=".db")
#     engine = create_engine(
#         f"sqlite:///{dbpath}",
#         isolation_level="AUTOCOMMIT",
#     )

#     # The agent has access to the SQL database through these tools
#     db = SQLDatabase(engine)
#     toolkit = SQLDatabaseToolkit(db=db, llm=model)
#     # Filter out sql_db_list_tables since it only lists the initial tables
#     tools = [tool for tool in toolkit.get_tools() if tool.name != "sql_db_list_tables"]

#     # Use the prebuilt ReAct agent graph
#     # https://langchain-ai.github.io/langgraph/agents/agents/
#     agent = create_react_agent(
#         model, tools, checkpointer=checkpointer, prompt=SYSTEM_PROMPT
#     )
#     config: RunnableConfig = {
#         "configurable": {"thread_id": "default"},
#         "recursion_limit": recursion_limit,
#     }

#     print_markdown(INTRO_TEXT.format(dbpath=dbpath))

#     while True:
#         # Accept input from the user
#         try:
#             prompt_txt = ask_prompt()
#         except (EOFError, KeyboardInterrupt):
#             print_markdown("Exiting...")
#             break

#         if not prompt_txt:
#             continue
#         prompt = HumanMessage(prompt_txt)

#         with console.status("Agent is thinking"):
#             # [START opentelemetry_adk_agent_span]
#             # Invoke the agent within a span
#             with tracer.start_as_current_span("invoke agent"):
#                 result = agent.invoke({"messages": [prompt]}, config=config)
#             # [END opentelemetry_adk_agent_span]

#         # Print history
#         render_messages(result["messages"])
