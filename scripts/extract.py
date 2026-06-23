import re
import time
from pathlib import Path
from urllib.parse import urljoin
import json

import requests
import yaml


# Project root is the folder above scripts/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(relative_path):
    """Convert a project-relative path into an absolute path."""
    return PROJECT_ROOT / relative_path


def build_ollama_url(base_url, endpoint):
    """Build the Ollama request URL from settings.yaml values."""
    base_url = str(base_url).rstrip("/") + "/"
    endpoint = str(endpoint).lstrip("/")

    return urljoin(base_url, endpoint)


def build_prompt(prompt_template, note_text, vocabulary, csv_columns, settings):
    """
    Replace prompt placeholders with the current note, vocabulary, and CSV template.
    """
    prompt_settings = settings.get("prompt", {})

    csv_template = json.dumps(csv_columns, indent=2, ensure_ascii=False)

    if prompt_settings.get("include_vocabulary", True):
        vocabulary_text = yaml.safe_dump(
            vocabulary,
            sort_keys=False,
            allow_unicode=True,
        )

        max_vocabulary_characters = prompt_settings.get(
            "max_vocabulary_characters",
            18000,
        )

        vocabulary_text = vocabulary_text[:max_vocabulary_characters]
    else:
        vocabulary_text = "Vocabulary not included."

    prompt = prompt_template.replace(
        "{{CSV_COLUMNS_TEMPLATE}}",
        csv_template,
    )
    prompt = prompt.replace(
        "{{VOCABULARY}}",
        vocabulary_text,
    )
    prompt = prompt.replace(
        "{{NOTE_TEXT}}",
        note_text,
    )

    return prompt


def clean_json_text(raw_text):
    """
    Remove common markdown wrappers before parsing a model response as JSON.
    """
    if not isinstance(raw_text, str):
        raise ValueError("Ollama returned a non-text response.")

    cleaned_text = raw_text.strip()

    if cleaned_text.startswith("```"):
        cleaned_text = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned_text,
            flags=re.IGNORECASE,
        )
        cleaned_text = re.sub(
            r"\s*```$",
            "",
            cleaned_text,
        )

    return cleaned_text.strip()


def parse_model_json(raw_text):
    """
    Parse one JSON object returned by Ollama.

    Ollama JSON mode should return valid JSON directly. The fallback searches
    for the first JSON object if extra text was returned.
    """
    cleaned_text = clean_json_text(raw_text)

    try:
        parsed = json.loads(cleaned_text)
    except json.JSONDecodeError:
        first_brace = cleaned_text.find("{")
        last_brace = cleaned_text.rfind("}")

        if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
            raise ValueError("Ollama response does not contain a JSON object.")

        try:
            parsed = json.loads(cleaned_text[first_brace:last_brace + 1])
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Ollama returned invalid JSON: {error}"
            ) from error

    if not isinstance(parsed, dict):
        raise ValueError("Ollama JSON output must be one object.")

    return parsed


def save_raw_response(note_id, raw_response, settings):
    """Save one raw Ollama response for debugging when enabled."""
    logging_settings = settings.get("logging", {})

    if not logging_settings.get("save_raw_model_response", True):
        return

    response_folder = project_path(
        logging_settings.get(
            "raw_response_folder",
            "logs/raw_responses",
        )
    )
    response_folder.mkdir(parents=True, exist_ok=True)

    safe_note_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(note_id))
    response_path = response_folder / f"{safe_note_id}_response.txt"

    with response_path.open("w", encoding="utf-8") as file:
        file.write(raw_response)


def build_request_payload(prompt, settings):
    """Build the POST body for Ollama's /api/generate endpoint."""
    ollama_settings = settings.get("ollama", {})

    payload = {
        "model": ollama_settings.get("model", "llama3.1:8b"),
        "prompt": prompt,
        "stream": ollama_settings.get("stream", False),
        "format": ollama_settings.get("format", "json"),
        "options": {
            "temperature": ollama_settings.get("temperature", 0),
            "num_predict": ollama_settings.get("num_predict", 4096),
            "num_ctx": ollama_settings.get("num_ctx", 8192),
        },
    }

    return payload


def request_ollama(prompt, settings):
    """
    Send the prompt to Ollama and return the text from the response field.
    """
    ollama_settings = settings.get("ollama", {})

    base_url = ollama_settings.get("base_url", "http://localhost:11434")
    endpoint = ollama_settings.get("endpoint", "/api/generate")
    timeout_seconds = ollama_settings.get("timeout_seconds", 180)

    url = build_ollama_url(base_url, endpoint)
    payload = build_request_payload(prompt, settings)

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            f"Could not connect to Ollama at {url}: {error}"
        ) from error

    try:
        response_json = response.json()
    except ValueError as error:
        raise ValueError(
            "Ollama returned a response that was not valid JSON."
        ) from error

    if response_json.get("error"):
        raise RuntimeError(
            f"Ollama returned an error: {response_json['error']}"
        )

    raw_response = response_json.get("response", "")

    if not raw_response:
        raise ValueError("Ollama returned an empty response.")

    return raw_response


def extract_note(note_text, note_id, settings, prompt_template, vocabulary, csv_columns):
    """
    Extract structured data from one office note using Ollama.

    Returns:
        A dictionary parsed from the JSON returned by the model.

    Raises:
        RuntimeError or ValueError when Ollama cannot return usable JSON.
    """
    prompt = build_prompt(
        prompt_template=prompt_template,
        note_text=note_text,
        vocabulary=vocabulary,
        csv_columns=csv_columns,
        settings=settings,
    )

    ollama_settings = settings.get("ollama", {})
    max_retries = ollama_settings.get("max_retries", 1)

    last_error = None

    for attempt_number in range(max_retries + 1):
        try:
            raw_response = request_ollama(prompt, settings)
            save_raw_response(note_id, raw_response, settings)
            return parse_model_json(raw_response)

        except (RuntimeError, ValueError) as error:
            last_error = error

            if attempt_number < max_retries:
                time.sleep(1)
                continue

    raise RuntimeError(
        f"Extraction failed for note_id={note_id}: {last_error}"
    )
