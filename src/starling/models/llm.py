import json
import logging
import os
import urllib.parse

import tiktoken
from openai.types.shared.reasoning_effort import ReasoningEffort
from pydantic_ai import models
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

from starling.models.retry_client import create_retrying_client

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_SETTINGS = {
    "openai-responses": OpenAIResponsesModelSettings(
        openai_previous_response_id="auto",
        openai_reasoning_summary="detailed",
        openai_reasoning_effort="medium",
    ),
    "openrouter": OpenRouterModelSettings(
        extra_body={"tool_choice": "auto"},
        openrouter_usage={
            "include": True,
        },
    ),
}


def get_pydantic_model(
    model_name: str,
    reasoning_effort: ReasoningEffort = "medium",
) -> models.Model:
    if model_name.startswith("custom:"):
        return _get_custom_model(
            model_name.removeprefix("custom:"), reasoning_effort=reasoning_effort, max_concurrency=4000
        )
    elif model_name.startswith("openai-responses:"):
        model = model_name.removeprefix("openai-responses:")
        settings = _DEFAULT_MODEL_SETTINGS["openai-responses"]
        provider = OpenAIProvider(http_client=create_retrying_client(max_concurrency=100))
        return OpenAIResponsesModel(model, settings=settings, provider=provider)
    elif model_name.startswith("openrouter:"):
        model = model_name.removeprefix("openrouter:")
        settings = _DEFAULT_MODEL_SETTINGS["openrouter"]
        provider = OpenRouterProvider(http_client=create_retrying_client(max_concurrency=100))
        return OpenRouterModel(model, settings=settings, provider=provider)
    elif model_name.startswith("google:"):
        return _get_google_model(model_name.removeprefix("google:"), max_concurrency=100)
    else:
        model = models.infer_model(model_name)

    return model


def _get_google_model(model_name: str, max_concurrency: int = 10) -> GoogleModel:
    """
    Create a Google Gemini model using service account credentials.

    Uses GCP_SERVICE_ACCOUNT_FILE env var for credentials and GCP_PROJECT_ID for project.
    Uses the global endpoint for Vertex AI (required for preview models like Gemini 3).
    Configures high thinking effort for Gemini 3 models by default, configurable via :effort=suffix.
    """
    # Lazy imports to avoid side effects when not using Google models
    from google.genai.types import ThinkingLevel
    from google.oauth2 import service_account
    from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
    from pydantic_ai.providers.google import GoogleProvider

    effort = "high"
    if ":effort=" in model_name:
        model_name, effort_str = model_name.split(":effort=")
        effort = effort_str.lower()

    service_account_file = os.getenv("GCP_SERVICE_ACCOUNT_FILE")
    project_id = os.getenv("GCP_PROJECT_ID")

    if not service_account_file:
        raise ValueError("GCP_SERVICE_ACCOUNT_FILE environment variable is required for Google models")
    if not project_id:
        raise ValueError("GCP_PROJECT_ID environment variable is required for Google models")

    credentials = service_account.Credentials.from_service_account_file(
        service_account_file,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    client = create_retrying_client(max_concurrency=max_concurrency)
    provider = GoogleProvider(credentials=credentials, project=project_id, location="global", http_client=client)

    if effort == "off":
        settings = GoogleModelSettings()
    else:
        try:
            level = getattr(ThinkingLevel, effort.upper())
        except AttributeError:
            logger.warning(f"Unknown thinking effort '{effort}', defaulting to HIGH")
            level = ThinkingLevel.HIGH

        settings = GoogleModelSettings(
            google_thinking_config={
                "thinking_level": level,
                "include_thoughts": True,
            }
        )

    return GoogleModel(model_name, provider=provider, settings=settings)


def _get_custom_model(
    model_spec: str,
    reasoning_effort: ReasoningEffort = "medium",
    max_concurrency: int = 1000,
) -> OpenAIChatModel:
    """Create an OpenAI-compatible chat model pointed at a custom endpoint."""
    client = create_retrying_client(max_concurrency=max_concurrency)

    model_name, explicit_endpoint = _parse_custom_model_spec(model_spec)
    endpoint = explicit_endpoint or _lookup_custom_endpoint(model_name)

    base_url = urllib.parse.urljoin(_normalize_endpoint(endpoint), "v1")

    provider = OpenAIProvider(
        base_url=base_url,
        api_key="your-api-key",
        http_client=client,
    )
    profile = OpenAIModelProfile(
        openai_supports_tool_choice_required=True,
        supports_json_object_output=True,
        supports_json_schema_output=True,
        native_output_requires_schema_in_instructions=False,
    )
    settings = OpenAIChatModelSettings(
        openai_reasoning_effort=reasoning_effort,
    )

    return OpenAIChatModel(
        model_name=model_name,
        provider=provider,
        profile=profile,
        settings=settings,
    )


def _parse_custom_model_spec(model_spec: str) -> tuple[str, str | None]:
    """Split a custom model spec into base model and optional endpoint override.

    Example accepted forms:
    - "moonshot/Kimi-K2-Thinking" (resolved via CUSTOM_ENDPOINTS_JSON)
    - "moonshot/Kimi-K2-Thinking@dgx001:32132" (inline host override)
    - "moonshot/Kimi-K2-Thinking@http://dgx001:32132" (explicit scheme)
    """

    if "@" not in model_spec:
        return model_spec, None

    model_name, endpoint = model_spec.split("@", 1)
    return model_name.strip(), endpoint.strip() or None


def _normalize_endpoint(endpoint: str) -> str:
    if not endpoint.startswith("http"):
        endpoint = f"http://{endpoint}"
    return endpoint.rstrip("/") + "/"


def _lookup_custom_endpoint(model_name: str) -> str:
    """Resolve a custom model name to an endpoint via the CUSTOM_ENDPOINTS_JSON env var.

    Example:
    `CUSTOM_ENDPOINTS_JSON='{"moonshot/Kimi-K2-Thinking":"http://dgx001:32132"}'`
    """
    raw_env = os.environ.get("CUSTOM_ENDPOINTS_JSON")
    if not raw_env:
        raise ValueError(
            f"No endpoint configured for custom model '{model_name}'. "
            "Pass an inline override (`custom:model@host:port`) or set CUSTOM_ENDPOINTS_JSON."
        )

    try:
        parsed_env = json.loads(raw_env)
    except json.JSONDecodeError as e:
        raise ValueError("Failed to parse CUSTOM_ENDPOINTS_JSON; expected a JSON object") from e

    if not isinstance(parsed_env, dict):
        raise ValueError("CUSTOM_ENDPOINTS_JSON must be a JSON object mapping model_name -> endpoint string")

    endpoint_map = {str(k): str(v) for k, v in parsed_env.items()}
    if model_name not in endpoint_map:
        raise ValueError(f"Unknown custom model: {model_name}")
    return endpoint_map[model_name]


def get_tokenizer(model_name: str = "o200k_base") -> tiktoken.Encoding:
    return tiktoken.get_encoding(model_name)
