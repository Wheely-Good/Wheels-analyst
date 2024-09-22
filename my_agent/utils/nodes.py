from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from my_agent.utils.tools import tools
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
import json
from typing_extensions import List, Dict, Any
from my_agent.utils.types import Turn, Suggestion
from importlib import resources
import my_agent.data

# lexis subgraph - look for cases of lexical mistakes and ways to improve the range of vocabulary used

# grammar subgraph - look for cases of grammatical mistakes and ways to improve the range of grammar structures used

# coherence subgraph (look for more natural ways to re-write the text)

# we want a data structure that shows the utterance in question where the mistake occurred, the context of it (for example the previous question), and then give a suggestion for an improvement as well as reason for why it's an improvement

class CoherenceState(TypedDict):
    interview: List[Dict[str, Any]]
    coherence_suggestions: List[Suggestion]

class GrammarState(TypedDict):
    interview: List[Dict[str, Any]]
    grammar_suggestions: List[Suggestion]

class LexisState(TypedDict):
    interview: List[Dict[str, Any]]
    lexis_suggestions: List[Suggestion]

def get_coherence_suggestions(state, config):
    suggestions = [Suggestion(context="coherence_test", utterance="coherence_test", suggestion="coherence_test", reason="coherence_test")]
    return {"coherence_suggestions": suggestions}


def get_lexis_suggestions(state, config):
    suggestions = [Suggestion(context="lexis_test", utterance="lexis_test", suggestion="lexis_test", reason="lexis_test")]
    return {"lexis_suggestions": suggestions}


def get_grammar_suggestions(state, config):
    suggestions = [Suggestion(context="grammar_test", utterance="grammar_test", suggestion="grammar_test", reason="grammar_test")]
    return {"grammar_suggestions": suggestions}


def turn_to_message(turn: Turn) -> dict:
    return {"role": "user" if turn["speaker"] == "examiner" else "assistant", "content": turn["text"]}

def add_interview(state, config):
    raw_interview = json.loads(resources.read_text(my_agent.data, "sample_1.json"))["dialogue"]["stage_1"]
    
    interview = [
        turn_to_message(Turn(speaker=turn["speaker"], text=turn["text"]))
        for turn in raw_interview
    ]
    
    return {"interview": interview}


coherence_builder = StateGraph(CoherenceState)

coherence_builder.add_node("get_coherence_suggestions", get_coherence_suggestions)
coherence_builder.set_entry_point("get_coherence_suggestions")
coherence_builder.add_edge("get_coherence_suggestions", END)


lexis_builder = StateGraph(LexisState)

lexis_builder.add_node("get_lexis_suggestions", get_lexis_suggestions)
lexis_builder.set_entry_point("get_lexis_suggestions")
lexis_builder.add_edge("get_lexis_suggestions", END)

grammar_builder = StateGraph(GrammarState)

grammar_builder.add_node("get_grammar_suggestions", get_grammar_suggestions)
grammar_builder.set_entry_point("get_grammar_suggestions")
grammar_builder.add_edge("get_grammar_suggestions", END)



# @lru_cache(maxsize=4)
# def _get_model(model_name: str):
#     if model_name == "openai":
#         model = ChatOpenAI(temperature=0, model_name="gpt-4o")
#     elif model_name == "anthropic":
#         model =  ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
#     else:
#         raise ValueError(f"Unsupported model type: {model_name}")

#     model = model.bind_tools(tools)
#     return model


# # Define the function that calls the model
# def call_model(state, config):
#     messages = state["messages"]
#     messages = [{"role": "system", "content": system_prompt}] + messages
#     model_name = config.get('configurable', {}).get("model_name", "anthropic")
#     model = _get_model(model_name)
#     response = model.invoke(messages)
#     # We return a list, because this will get added to the existing list
#     return {"messages": [response]}

# # Define the function to execute tools
# tool_node = ToolNode(tools)