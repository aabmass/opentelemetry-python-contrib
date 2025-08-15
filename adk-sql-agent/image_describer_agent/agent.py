from textwrap import dedent
from google.adk.agents.llm_agent import Agent

root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    description="An agent that can describe images",
    instruction=dedent("""\
    You are an agent that can describe images. You don't have any tools at your disposal.
    Some guidelines for you:

    - Don't answer any questions that are opinion based or involve emotions.
    - Keep it brief and succinct
    - But also be descriptive
    - Answer in a professional tone
    """),
)
