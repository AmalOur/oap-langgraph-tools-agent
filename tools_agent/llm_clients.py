import os
from openai import OpenAI

MODEL_REGISTRY = {
    "gcore/qwen3-30b-a3b": {
        "type": "chat",
        "base_url": "https://inference-instance-qwen3-30b-ust2hkbr.ai.gcore.dev/v1",
        "model": "Qwen/Qwen3-30B-A3B",
        "env_key": "QWEN_LLM_API_KEY",  
    },
    "gcore/gte-qwen2-1.5b": {
        "type": "embedding",
        "base_url": "https://inference-instance-gte-qwen2-1.5b-ust2hkbr.ai.gcore.dev/v1".replace("1.5b","1.5b"),
        "model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "env_key": "QWEN_EMBEDDINGS_API_KEY",
    },
}

def make_openai_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={"X-API-Key": api_key},
    )

def resolve_key(config, key_name: str) -> str:
    api_keys = (config.get("configurable") or {}).get("apiKeys") or {}
    return api_keys.get(key_name) or os.environ.get(key_name) or ""

def chat_complete(config, messages, temperature=0.7, max_tokens=2000):
    sel = (config.get("configurable") or {}).get("model_name", "gcore/qwen3-30b-a3b")
    spec = MODEL_REGISTRY[sel]
    client = make_openai_client(spec["base_url"], resolve_key(config, spec["env_key"]))
    resp = client.chat.completions.create(
        model=spec["model"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

def embed_text(config, text: str):
    sel = (config.get("configurable") or {}).get("embedding_model_name", "gcore/gte-qwen2-1.5b")
    spec = MODEL_REGISTRY[sel]
    client = make_openai_client(spec["base_url"], resolve_key(config, spec["env_key"]))
    resp = client.embeddings.create(model=spec["model"], input=text)
    return resp.data[0].embedding