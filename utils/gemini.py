from google import genai
from google.genai import types
import os

def gemini_available() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY"))

# Shared client (new SDK)
client = genai.Client()

def create_operator_chat():
    return client.chats.create(
        model="gemini-2.5-flash",
        history=[
            types.Content(
                role="system",
                parts=[types.Part.from_text(
                    "You are an AI assistant helping a subway worker record found items. "
                    "Always output the final answer as a SINGLE structured JSON object only. "
                )]
            )
        ]
    )

def create_user_chat():
    return client.chats.create(
        model="gemini-2.0-flash",
        history=[
            types.Content(
                role="system",
                parts=[types.Part.from_text(
                    "You help users report lost items. "
                    "Always output the final answer as a SINGLE structured JSON object only. "
                )]
            )
        ]
    )

def is_structured_record(text: str) -> bool:
    return text.strip().startswith("{") and text.strip().endswith("}")

def extract_field(text: str, field: str) -> str:
    import re
    pattern = rf"{field}\s*:\s*(.*)"
    m = re.search(pattern, text)
    return m.group(1).strip() if m else ""

def standardize_description(text: str, tags: dict):
    import json
    try:
        return json.loads(text)
    except:
        return None

