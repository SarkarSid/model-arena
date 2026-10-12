"""
Model configurations — driven by environment variables.
Add or remove models by setting/unsetting the corresponding env vars.
Each model needs: ENDPOINT, API_KEY, and MODEL_ID.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    key: str                # Internal key e.g. "gpt4o"
    display_name: str       # Shown in UI e.g. "GPT-4o"
    model_id: str           # API model identifier
    endpoint: str           # Base URL for API
    api_key: str            # API key / subscription key
    api_type: str           # "azure" | "openai_compat" | "apim"
    color: str              # Hex color for card border
    icon: str               # Emoji for card header
    api_version: str = "2024-05-01-preview"   # Azure only
    extra_headers: dict = field(default_factory=dict)
    max_tokens_limit: int = 4096


def load_models() -> list[ModelConfig]:
    """
    Load model configs from environment variables.
    Supports up to 6 model slots. Configure via Azure Web App settings.

    Required env vars per slot (replace N with 1-6):
      MODEL_N_NAME        = Display name
      MODEL_N_ID          = Model/deployment ID
      MODEL_N_ENDPOINT    = Base URL
      MODEL_N_API_KEY     = API key or APIM subscription key
      MODEL_N_TYPE        = azure | openai_compat | apim

    Optional:
      MODEL_N_API_VERSION = Azure API version (default: 2024-05-01-preview)
      MODEL_N_COLOR       = Hex border color (default per slot)
      MODEL_N_ICON        = Emoji icon
    """
    slot_colors = ["#0078d4", "#ff6900", "#107c10", "#b146c2", "#008575", "#d13438"]
    slot_icons  = ["🤖", "🧠", "⚡", "🔮", "🌟", "🚀"]

    models = []
    for i in range(1, 7):
        name = os.getenv(f"MODEL_{i}_NAME")
        if not name:
            continue
        model_id  = os.getenv(f"MODEL_{i}_ID", "gpt-4o")
        endpoint  = os.getenv(f"MODEL_{i}_ENDPOINT", "")
        api_key   = os.getenv(f"MODEL_{i}_API_KEY", "")
        api_type  = os.getenv(f"MODEL_{i}_TYPE", "apim").lower()
        color     = os.getenv(f"MODEL_{i}_COLOR", slot_colors[i - 1])
        icon      = os.getenv(f"MODEL_{i}_ICON", slot_icons[i - 1])
        api_ver   = os.getenv(f"MODEL_{i}_API_VERSION", "2024-05-01-preview")

        models.append(ModelConfig(
            key=f"model_{i}",
            display_name=name,
            model_id=model_id,
            endpoint=endpoint,
            api_key=api_key,
            api_type=api_type,
            color=color,
            icon=icon,
            api_version=api_ver,
        ))

    # Fall back to demo models if nothing configured
    if not models:
        models = _demo_models()
    return models


def _demo_models() -> list[ModelConfig]:
    """Hardcoded demo models — replace with env vars in production."""
    return [
        ModelConfig(
            key="gpt4o",
            display_name="GPT-4o (Azure)",
            model_id=os.getenv("GPT4O_DEPLOYMENT", "gpt-4o"),
            endpoint=os.getenv("APIM_ENDPOINT", "https://your-apim.azure-api.net/openai"),
            api_key=os.getenv("APIM_SUBSCRIPTION_KEY", ""),
            api_type="apim",
            color="#0078d4",
            icon="🤖",
        ),
        ModelConfig(
            key="mistral",
            display_name="Mistral (On-Prem)",
            model_id=os.getenv("MISTRAL_MODEL_ID", "mistral-large-latest"),
            endpoint=os.getenv("MISTRAL_ENDPOINT", "http://mistral-server:8000/v1"),
            api_key=os.getenv("MISTRAL_API_KEY", "not-required"),
            api_type="openai_compat",
            color="#ff6900",
            icon="🔥",
        ),
    ]
