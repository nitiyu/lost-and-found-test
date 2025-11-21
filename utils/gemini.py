import google.generativeai as genai
from google.generativeai import types
import streamlit as st

MODEL_NAME = "gemini-2.5-flash"


def gemini_available() -> bool:
    return bool(st.secrets.get("GEMINI_API_KEY"))


def get_gemini_client():
    key = st.secrets.get("GEMINI_API_KEY")
    if not key:
        return None
    return genai.Client(api_key=key)


def create_operator_chat():
    client = get_gemini_client()
    if client is None:
        raise RuntimeError("Gemini not configured")
    return client.chats.create(model=MODEL_NAME, config=types.GenerateContentConfig(system_instruction=st.secrets.get("GENERATOR_SYSTEM_PROMPT")))


def create_user_chat():
    client = get_gemini_client()
    if client is None:
        raise RuntimeError("Gemini not configured")
    return client.chats.create(model=MODEL_NAME, config=types.GenerateContentConfig(system_instruction=st.secrets.get("USER_SIDE_GENERATOR_PROMPT")))


def is_structured_record(message: str) -> bool:
    return message.strip().startswith("Subway Location:")


def standardize_description(text: str, tags: dict) -> dict:
    # Expect STANDARDIZER_PROMPT in secrets (or you can hardcode it in secrets)
    client = get_gemini_client()
    if client is None:
        raise RuntimeError("Gemini not configured")
    system = st.secrets.get("STANDARDIZER_PROMPT")
    tags_summary = (
        "\n--- TAGS REFERENCE ---\n"
        f"Subway Location tags: {', '.join(tags['locations'][:50])}\n"
        f"Color tags: {', '.join(tags['colors'][:50])}\n"
        f"Item Category tags: {', '.join(tags['categories'][:50])}\n"
        f"Item Type tags: {', '.join(tags['item_types'][:50])}\n"
    )
    full = system + "\n\nHere is the structured input to standardize:\n" + text + tags_summary
    resp = client.models.generate_content(model=MODEL_NAME, contents=full)
    cleaned = resp.text.strip()
    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}") + 1
    json_text = cleaned[json_start:json_end]
    import json as _json
    data = _json.loads(json_text)
    from datetime import datetime, timezone
    if "time" not in data or not data["time"]:
        data["time"] = datetime.now(timezone.utc).isoformat()
    for key in ["subway_location","color","item_type"]:
        if key in data and isinstance(data[key], str):
            data[key] = [data[key]]
        elif key not in data:
            data[key] = []
    if "item_category" not in data:
        data["item_category"] = "null"
    if "description" not in data:
        data["description"] = ""
    return data


def extract_field(text: str, field: str) -> str:
    import re
    m = re.search(rf"{field}:\s*(.*)", text)
    return m.group(1).strip() if m else "null"