# tools_agent/nodes.py
from typing import TypedDict
from tools_agent.llm_clients import chat_complete

class State(TypedDict):
    messages: list

def chat(state: State, config) -> State:
    text = chat_complete(
        config=config,
        messages=state["messages"],
        temperature=(config["configurable"].get("temperature") or 0.7),
        max_tokens=(config["configurable"].get("max_tokens") or 2000),
    )
    return {"messages": [*state["messages"], {"role": "assistant", "content": text}]}
