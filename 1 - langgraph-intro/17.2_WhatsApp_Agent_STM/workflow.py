from typing import Annotated, Optional, TypedDict
import sqlite3

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph .graph import StateGraph, END, add_messages

import os
import logging
from dotenv import load_dotenv

# Configurations
wf_logger = logging.getLogger("WhatsApp.Graph")
agent_logger = logging.getLogger("WhatsApp.Agent")

load_dotenv(override=True)

# Initialize LLMs (Using Groq API as configured previously)
def get_groq_llm(model_name="openai/gpt-oss-20b"):
    agent_logger.info(f"Initializing LLM: {model_name}")
    return ChatOpenAI(
        model=model_name,
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        max_tokens=1000
    )

llm = get_groq_llm()
vision_llm = get_groq_llm("meta-llama/llama-4-scout-17b-16e-instruct")

# Database connection for memory
wf_logger.info("Connecting to SQLite Memory Database...")
sqlite_conn = sqlite3.connect(
    "whatsapp_bot_memory.sqlite",
    check_same_thread=False
)
memory = SqliteSaver(sqlite_conn)

# Initial State for the WhatsApp Agent
class WhatsAppState(TypedDict):
    messages: Annotated[list, add_messages]
    image_url: Optional[str]


def chat_agent(state: WhatsAppState):
    user_msg = state["messages"][-1].content
    agent_logger.info(f"[Chat Agent] Triggered. Processing User Message: '{user_msg}'")

    response = llm.invoke(state["messages"])

    agent_logger.info(f"[Chat Agent] Generated Reply: '{response.content}'")
    return {"messages": [response]}


def vision_agent(state: WhatsAppState):
    last_user_message = state["messages"][-1].content
    agent_logger.info(f"[Vision Agent] Triggered. Processing Image + Prompt: '{last_user_message}'")
    agent_logger.info(f"[Vision Agent] Image URL/Base64 length: {len(state['image_url'])} chars")

    response = vision_llm.invoke([
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": state["image_url"]}},
                {"type": "text", "text": last_user_message or "Describe this image"}
            ]
        }
    ])

    agent_logger.info(f"[Vision Agent] Generated Reply: '{response.content}'")
    return {"messages": [response]}


def supervisor_agent(state: WhatsAppState):
    wf_logger.info("--- Supervisor Routing Decision ---")
    image_data = state.get("image_url")

    if image_data:
        wf_logger.info(f"Decision: Found image_url. Routing to 'vision_agent'")
        return "vision_agent"

    wf_logger.info(f"Decision: No image detected. Routing to 'chat_agent'")
    return "chat_agent"


def build_langgraph_app():
    wf_logger.info("Building and compiling LangGraph State Machine...")
    graph = StateGraph(WhatsAppState)

    graph.add_node("chat_agent", chat_agent)
    graph.add_node("vision_agent", vision_agent)

    graph.set_conditional_entry_point(
        supervisor_agent,
        {
            "chat_agent": "chat_agent",
            "vision_agent": "vision_agent"
        }
    )

    graph.add_edge("chat_agent", END)
    graph.add_edge("vision_agent", END)

    langgraph_app = graph.compile(checkpointer=memory)
    wf_logger.info("Graph compiled successfully!")
    return langgraph_app