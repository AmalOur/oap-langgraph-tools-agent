import os
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import StructuredTool

from tools_agent.utils.tools import (
    create_rag_tool,
    wrap_mcp_authenticate_tool,
    create_langchain_mcp_tool,
)
from tools_agent.utils.token import fetch_tokens
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession


UNEDITABLE_SYSTEM_PROMPT = (
    "\nIf the tool throws an error requiring authentication, provide the user with a "
    "Markdown link to the authentication page and prompt them to authenticate."
)
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that has access to a variety of tools."


# -------------------- UI CONFIG SCHEMAS (no runtime calls here) --------------------
class RagConfig(BaseModel):
    rag_url: Optional[str] = None
    collections: Optional[List[str]] = None


class MCPConfig(BaseModel):
    url: Optional[str] = Field(default=None, optional=True)
    tools: Optional[List[str]] = Field(default=None, optional=True)
    auth_required: Optional[bool] = Field(default=False, optional=True)


class GraphConfigPydantic(BaseModel):
    # IMPORTANT: use a provider prefix that your helpers recognize, e.g. "gcore:"
    model_name: Optional[str] = Field(
        default="Qwen/Qwen3-30B-A3B",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "label": "Chat model",
                "description": "OpenAI-compatible chat model",
                "default": "Qwen/Qwen3-30B-A3B",
                "options": [
                    {"label": "Qwen3-30B-A3B (Gcore)", "value": "Qwen/Qwen3-30B-A3B"},
                ],
            }
        },
    )
    temperature: Optional[float] = Field(
        default=0.7,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 0.7,
                "min": 0,
                "max": 2,
                "step": 0.1,
                "description": "Controls randomness (0 = deterministic, 2 = creative)",
            }
        },
    )
    max_tokens: Optional[int] = Field(
        default=4000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 4000,
                "min": 1,
                "description": "The maximum number of tokens to generate",
            }
        },
    )
    system_prompt: Optional[str] = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        metadata={
            "x_oap_ui_config": {
                "type": "textarea",
                "placeholder": "Enter a system prompt...",
                "description": (
                    "The system prompt to use in all generations. The following prompt "
                    "will always be included at the end of the system prompt:\n"
                    f"---{UNEDITABLE_SYSTEM_PROMPT}\n---"
                ),
                "default": DEFAULT_SYSTEM_PROMPT,
            }
        },
    )
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={"x_oap_ui_config": {"type": "mcp"}},
    )
    rag: Optional[RagConfig] = Field(
        default=None,
        optional=True,
        metadata={"x_oap_ui_config": {"type": "rag"}},
    )
    # Allow OAP Settings → apiKeys to be forwarded (optional)
    apiKeys: Optional[Dict[str, str]] = Field(default=None)


# -------------------- RUNTIME HELPERS (safe to call in graph()) --------------------
def get_api_key_for_model(model_name: str, config: RunnableConfig):
    model_name_l = (model_name or "").lower()
    model_to_key = {
        "openai:": "OPENAI_API_KEY",
        "anthropic:": "ANTHROPIC_API_KEY",
        "google:": "GOOGLE_API_KEY",
        "gcore:": "QWEN_LLM_API_KEY",   # <- for your Gcore chat endpoint
    }
    key_name = next((key for prefix, key in model_to_key.items()
                     if model_name_l.startswith(prefix)), None)
    if not key_name:
        return None
    api_keys = (config.get("configurable", {}) or {}).get("apiKeys", {}) or {}
    if api_keys.get(key_name):
        return api_keys[key_name]
    return os.getenv(key_name)


def get_connection_overrides(model_name: str, config: RunnableConfig):
    """
    Return (base_url, default_headers) for an OpenAI-compatible server, else (None, None).
    """
    model_name_l = (model_name or "").lower()
    if model_name_l.startswith("gcore:"):
        base_url = "https://inference-instance-qwen3-30b-ust2hkbr.ai.gcore.dev/v1"
        api_key = get_api_key_for_model(model_name, config) or ""
        default_headers = {"X-API-Key": api_key} if api_key else None
        return base_url, default_headers
    return None, None

def resolve_model_and_provider(model_name: str):
    """
    Returns (clean_model_name, model_provider) for init_chat_model.
    We keep our own prefixes for selection, but we give init_chat_model
    what it expects.
    """
    if not model_name:
        return None, None
    lower = model_name.lower()

    # Your OpenAI-compatible Gcore endpoint
    if lower.startswith("gcore:"):
        return model_name.split(":", 1)[1], "openai"

    # Native providers (examples)
    if lower.startswith("openai:"):
        return model_name.split(":", 1)[1], "openai"
    if lower.startswith("anthropic:"):
        return model_name.split(":", 1)[1], "anthropic"
    if lower.startswith("google:"):
        return model_name.split(":", 1)[1], "google-genai"

    # No prefix → if it’s an OpenAI-compatible custom, pass provider explicitly
    return model_name, None

# -------------------- GRAPH FACTORY --------------------
async def graph(config: RunnableConfig):
    cfg = GraphConfigPydantic(**(config.get("configurable", {}) or {}))
    tools: List[StructuredTool] = []

    # RAG tools (optional)
    supabase_token = (config.get("configurable", {}) or {}).get("x-supabase-access-token")
    if cfg.rag and cfg.rag.rag_url and cfg.rag.collections and supabase_token:
        for collection in cfg.rag.collections:
            rag_tool = await create_rag_tool(cfg.rag.rag_url, collection, supabase_token)
            tools.append(rag_tool)

    # MCP tools (optional)
    if cfg.mcp_config and cfg.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None

    if (
        cfg.mcp_config
        and cfg.mcp_config.url
        and cfg.mcp_config.tools
        and (mcp_tokens or not cfg.mcp_config.auth_required)
    ):
        server_url = cfg.mcp_config.url.rstrip("/") + "/mcp"
        tool_names_to_find = set(cfg.mcp_config.tools)
        fetched_mcp_tools_list: List[StructuredTool] = []
        names_of_tools_added = set()
        headers = (
            {"Authorization": f"Bearer {mcp_tokens['access_token']}"}
            if mcp_tokens is not None
            else None
        )
        try:
            async with streamablehttp_client(server_url, headers=headers) as streams:
                read_stream, write_stream, _ = streams
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    page_cursor = None
                    while True:
                        tool_list_page = await session.list_tools(cursor=page_cursor)
                        if not tool_list_page or not tool_list_page.tools:
                            break
                        for mcp_tool in tool_list_page.tools:
                            if not tool_names_to_find or (
                                mcp_tool.name in tool_names_to_find
                                and mcp_tool.name not in names_of_tools_added
                            ):
                                lc_tool = create_langchain_mcp_tool(
                                    mcp_tool, mcp_server_url=server_url, headers=headers
                                )
                                fetched_mcp_tools_list.append(
                                    wrap_mcp_authenticate_tool(lc_tool)
                                )
                                if tool_names_to_find:
                                    names_of_tools_added.add(mcp_tool.name)
                        page_cursor = tool_list_page.nextCursor
                        if not page_cursor:
                            break
                        if tool_names_to_find and len(names_of_tools_added) == len(tool_names_to_find):
                            break
                    tools.extend(fetched_mcp_tools_list)
        except Exception as e:
            print(f"Failed to fetch MCP tools: {e}")

    # ---- build chat model with overrides ----
    api_key = get_api_key_for_model(cfg.model_name, config) or "No token found"
    base_url, default_headers = get_connection_overrides(cfg.model_name, config)

    clean_model, provider = resolve_model_and_provider(cfg.model_name)

    kwargs = dict(
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers,
    )

    # If provider can’t be inferred, pass it explicitly.
    # For your Gcore case, provider will be "openai".
    if provider:
        kwargs["model_provider"] = provider
    else:
        # still ambiguous? default to openai if you know it’s OpenAI-compatible
        kwargs["model_provider"] = "openai"

    model = init_chat_model(clean_model, **kwargs)