"""Import alias for grok_provider. Not listed in UI (ProviderManager skips this file).

Legacy model_type values `xai` / `xai_provider` normalize to `grok_provider` in the worker.
"""
from model_providers.grok_provider import *  # noqa: F401,F403
from model_providers.grok_provider import (  # noqa: F401
    connect,
    disconnect,
    ask_model,
    ask_model_chat,
    create_embeddings,
    token_limit,
    emb_token_limit,
    do_chat_construct,
    native_func_call,
    tags,
)
