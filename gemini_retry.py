import os
import time
from typing import Any


# Gemini can return 503 when a specific model is overloaded. Retrying the same
# model helps briefly, but production-style reliability needs model fallback.
DEFAULT_PRIMARY_MODEL = "gemini-2.5-flash"
DEFAULT_FALLBACK_MODELS = ["gemini-2.5-flash-lite", "gemini-flash-latest"]


def get_model_sequence(primary_model: str = DEFAULT_PRIMARY_MODEL) -> list[str]:
    """Return primary model plus configurable fallbacks without duplicates.

    Override fallbacks in .env if needed:
    GEMINI_FALLBACK_MODELS=gemini-2.5-flash-lite,gemini-flash-latest
    """
    fallback_text = os.getenv(
        "GEMINI_FALLBACK_MODELS",
        ",".join(DEFAULT_FALLBACK_MODELS),
    )
    requested_models = [primary_model]
    requested_models.extend(
        model.strip()
        for model in fallback_text.split(",")
        if model.strip()
    )

    models = []
    for model in requested_models:
        if model not in models:
            models.append(model)

    return models


def is_transient_gemini_error(error: Exception) -> bool:
    """Return True for overload/service errors where fallback is useful."""
    message = str(error).lower()
    transient_terms = [
        "503",
        "500",
        "502",
        "504",
        "unavailable",
        "overloaded",
        "high demand",
        "service unavailable",
        "temporarily",
    ]
    return any(term in message for term in transient_terms)


def generate_content_with_fallback(
    *,
    client: Any,
    contents: list[Any],
    config: Any,
    primary_model: str = DEFAULT_PRIMARY_MODEL,
    attempts_per_model: int = 2,
    base_delay_seconds: float = 1.0,
    logger: Any = None,
    operation_name: str = "Gemini request",
) -> tuple[Any, str]:
    """Call Gemini with retries and fallback models.

    Non-transient errors, such as blocked API keys or permission failures, are
    raised immediately because trying another model will not fix them.
    """
    last_error = None

    for model in get_model_sequence(primary_model):
        for attempt in range(1, attempts_per_model + 1):
            try:
                if logger:
                    logger.info("%s using model=%s attempt=%s", operation_name, model, attempt)

                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                return response, model
            except Exception as error:
                last_error = error

                if not is_transient_gemini_error(error):
                    raise

                if logger:
                    logger.warning(
                        "%s failed on model=%s attempt=%s: %s",
                        operation_name,
                        model,
                        attempt,
                        error,
                    )

                delay = base_delay_seconds * (2 ** (attempt - 1))
                time.sleep(delay)

        if logger:
            logger.warning(
                "%s switching model after repeated transient failures: %s",
                operation_name,
                model,
            )

    raise RuntimeError(
        f"{operation_name} failed after model fallbacks: {last_error}"
    )
