"""
Unified model calling layer.
Handles Azure APIM, native Azure OpenAI, and OpenAI-compatible endpoints.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from openai import AzureOpenAI, OpenAI
from config import ModelConfig


@dataclass
class ModelResponse:
    model_key: str
    display_name: str
    content: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def tokens_per_second(self) -> float:
        if self.latency_ms == 0:
            return 0.0
        return round(self.completion_tokens * 1000 / self.latency_ms, 1)


def call_model(
    config: ModelConfig,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> ModelResponse:
    """Call a model and return a normalised ModelResponse."""
    start = time.time()
    try:
        client = _build_client(config)
        response = client.chat.completions.create(
            model=config.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = int((time.time() - start) * 1000)
        usage = response.usage
        return ModelResponse(
            model_key=config.key,
            display_name=config.display_name,
            content=response.choices[0].message.content or "",
            latency_ms=latency_ms,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )
    except Exception as exc:
        latency_ms = int((time.time() - start) * 1000)
        return ModelResponse(
            model_key=config.key,
            display_name=config.display_name,
            content="",
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error=str(exc),
        )


def _build_client(config: ModelConfig) -> OpenAI | AzureOpenAI:
    """
    Build the right OpenAI SDK client for the model type.

    - "azure"         : AzureOpenAI — key auth if MODEL_N_API_KEY is set,
                        otherwise Managed Identity (DefaultAzureCredential)
    - "apim"          : AzureOpenAI via APIM — subscription key in
                        Ocp-Apim-Subscription-Key header if set,
                        otherwise Managed Identity
    - "openai_compat" : Generic OpenAI-compatible endpoint (vLLM, Ollama,
                        Mistral, etc.) — api_key optional
    """
    if config.api_type in ("azure", "apim"):
        if not config.api_key:
            # No key supplied → authenticate via Managed Identity.
            # On Azure Web App this uses the app's system-assigned identity;
            # locally it falls through to az CLI / env credentials.
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default",
            )
            return AzureOpenAI(
                azure_endpoint=config.endpoint,
                azure_ad_token_provider=token_provider,
                api_version=config.api_version,
                default_headers=config.extra_headers or None,
            )

        if config.api_type == "azure":
            return AzureOpenAI(
                azure_endpoint=config.endpoint,
                api_key=config.api_key,
                api_version=config.api_version,
            )

        # apim with subscription key — key goes in Ocp-Apim-Subscription-Key;
        # a non-empty api_key is required by the SDK so we reuse the same value.
        return AzureOpenAI(
            azure_endpoint=config.endpoint,
            api_key=config.api_key,
            api_version=config.api_version,
            default_headers={
                "Ocp-Apim-Subscription-Key": config.api_key,
                **config.extra_headers,
            },
        )

    # openai_compat — works with vLLM, Ollama (/v1), Mistral server, LM Studio, etc.
    return OpenAI(
        base_url=config.endpoint,
        api_key=config.api_key or "not-required",
        default_headers=config.extra_headers or None,
    )
