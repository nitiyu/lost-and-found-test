from google import genai
from google.genai import types
import streamlit as st
import json as _json
from datetime import datetime, timezone

MODEL_NAME = "gemini-2.5-flash"


# -----------------------------
# Gemini API initialization
# -----------------------------
def gemini_available() -> bool:
    return bool(st.secrets.get("GEMINI_API_KEY"))


def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


# -----------------------------
# Operator Chat
# -----------------------------
def create_operator_chat():
    client = get_gemini_client()
    if client is None:
        raise RuntimeError("Gemini not configured")

    system_prompt = st.secrets.get("GENERATOR_SYSTEM_PROMPT", "")

    def send(message):
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Content(role="system", parts=[types.Part.from_text(system_prompt)]),
                types.Content(role="user", parts=[types.Part.from_text(message)])
            ]
        )
        return response.text

    return send


# -----------------------------
# User Chat
# -----------------------------
def create_user_chat():
    client = get_gemini_client()
    if client is None:
        raise RuntimeError("Gemini not configured")

    system_prompt = st.secrets.get("USER_SIDE_GENERATOR_PROMPT", "")

    def send(message):
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Content(role="system", parts=[types.Part.from_text(system_prompt)]),
                types.Content(role="user", parts=[types.Part.from_text(message)])
            ]
        )
        return response.text

    return send


# -----------------------------
# Check structured record pattern
# -----------------------------
def is_structured_record(message: str) -> bool:
    return message.strip().startswith("Subway Location:")


# -----------------------------
# Standardize description into JSON
# -----------------------------
def standardize_description(text: str, tags: dict) -> dict:
    client = get_gemini_client()
    if client is None:
        raise RuntimeError("Gemini not configured")

    system = st.secrets.get("STANDARDIZER_PROMPT", "")

    tags_summary = (
        "\n--- TAGS REFERENCE ---\n"
        f"Subway Location tags: {', '.join(tags['locations'][:50])}\n"
        f"Color tags: {', '.join(tags['colors'][:50])}\n"
        f"Item Category tags: {', '.join(tags['categories'][:50])}\n"
        f"Item Type tags: {', '.join(tags['item_types'][:50])}\n"
    )

    full_prompt = (
        system
        + "\n\nHere is the structured input to standardize:\n"
        + text
        + tags_summary
    )

    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Content(role="user", parts=[types.Part.from_text(full_prompt)])]
    )

    cleaned = resp.text.strip()

    # Extract JSON
    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}") + 1
    json_text = cleaned[json_start:json_end]

    data = _json.loads(json_text)

    # Insert or fix time
    if "time" not in data or not data["time"]:
        data["time"] = datetime.now(timezone.utc).isoformat()

    # Ensure list fields exist
    for key in ["subway_location", "color", "item_type"]:
        if key in data and isinstance(data[key], str):
            data[key] = [data[key]]
        elif key not in data:
            data[key] = []

    # Default fallback values
    if "item_category" not in data:
        data["item_category"] = "null"
    if "description" not in data:
        data["description"] = ""

    return data


# -----------------------------
# Extract a field from structured text
# -----------------------------
def extract_field(text: str, field: str) -> str:
    import re
    m = re.search(rf"{field}:\s*(.*)", text)
    return m.group(1).strip() if m else "null"
