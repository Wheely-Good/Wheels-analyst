from typing_extensions import TypedDict, Literal, List
from typing import Annotated, Optional

from langgraph.graph import StateGraph, END, add_messages
from my_agent.utils.state import AgentState
from my_agent.utils.nodes import add_interview, coherence_builder, grammar_builder, lexis_builder
from my_agent.utils.types import Turn, Suggestion


# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]


class Turn(TypedDict):
    speaker: Literal["examiner", "candidate"]
    text: str

class EntryGraphState(TypedDict):
    interview: Annotated[list[Turn], add_messages]
    coherence_suggestions: List[Suggestion]
    grammar_suggestions: List[Suggestion]
    lexis_suggestions: List[Suggestion]

# Define a new graph
main_analyst = StateGraph(EntryGraphState, config_schema=GraphConfig)

# add the interview state 
main_analyst.add_node("add_interview", add_interview)

# add the coherence, grammar, and lexis analysis agents as nodes
main_analyst.add_node("coherence_analyst", coherence_builder.compile())
main_analyst.add_node("grammar_analyst", grammar_builder.compile())
main_analyst.add_node("lexis_analyst", lexis_builder.compile())

# set the entry point to add the interview state
main_analyst.set_entry_point("add_interview")

# add the edges from add_interview to the coherence, grammar, and lexis analysis agents
main_analyst.add_edge("add_interview", "coherence_analyst")
main_analyst.add_edge("add_interview", "grammar_analyst")
main_analyst.add_edge("add_interview", "lexis_analyst")
main_analyst.add_edge("coherence_analyst", END)
main_analyst.add_edge("grammar_analyst", END)
main_analyst.add_edge("lexis_analyst", END)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = main_analyst.compile()
