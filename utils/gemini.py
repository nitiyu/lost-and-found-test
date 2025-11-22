import google.generativeai as genai
from google.generativeai import types
import streamlit as st
import json as _json
from datetime import datetime, timezone


MODEL_NAME = "gemini-2.5-flash"


# -----------------------------
# Gemini API initialization
# -----------------------------
def gemini_available() -> bool:
    return bool(st.secrets.get("GEMINI_API_KEY"))


def get_gemini_model():
    """Returns a configured GenerativeModel instance."""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return None
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)


# -----------------------------
# Operator Chat
# -----------------------------
def create_operator_chat():
    model = get_gemini_model()
    if model is None:
        raise RuntimeError("Gemini not configured")

    system_prompt = st.secrets.get("GENERATOR_SYSTEM_PROMPT", "")

    # Return a function that behaves like a chat message interface
    def send(message):
        response = model.generate_content(
            contents=[
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": message}
            ]
        )
        return response.text

    return send


# -----------------------------
# User Chat
# -----------------------------
def create_user_chat():
    model = get_gemini_model()
    if model is None:
        raise RuntimeError("Gemini not configured")

    system_prompt = st.secrets.get("USER_SIDE_GENERATOR_PROMPT", "")

    def send(message):
        response = model.generate_content(
            contents=[
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": message}
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
    model = get_gemini_model()
    if model is None:
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

    # Generate
    resp = model.generate_content(full_prompt)
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
