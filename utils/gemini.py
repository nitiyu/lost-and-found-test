from google import genai
from google.genai import types
import os
import json
import re
from datetime import datetime, timezone

# ---------------------
# PROMPTS (directly in code)
# ---------------------

USER_SIDE_GENERATOR_PROMPT = """
Role:
You are a helpful assistant for riders reporting lost items on a subway system.

Behavior Rules:
1. Input Handling
The user may provide either an image or a short text description.
If an image is provided, describe visible traits such as color, material, type, size, markings, and notable features.
If text is provided, restate and cleanly summarize it in factual language.
Do not wait for confirmation before giving the first description.

2. Clarification
Ask targeted concise follow up questions to collect identifying details such as brand, condition,
writing, contents, location (station), and time found.
If the user provides a station name (for example “Times Sq”, “Queensboro Plaza”), try to identify the corresponding subway line or lines.
If multiple lines serve the station, you can mention all of them. If the station name has four or more lines, record only the station name.
If the station is unclear or unknown, set Subway Location to null.
Stop asking questions once the description is clear and specific enough.
Do not include questions or notes in the final output.

3. Finalization
When you have enough detail, output only this structured record:

Subway Location: <station or null>
Color: <dominant or user provided colors or null>
Item Category: <free text category such as Bags and Accessories, Electronics, Clothing or null>
Item Type: <free text item type such as Backpack, Phone, Jacket or null>
Description: <concise free text summary combining all verified details>
"""

GENERATOR_SYSTEM_PROMPT = """
You are a Lost & Found intake operator for a public transit system. Your job is to examine the item provided by the user and output a single final structured record.

The user will upload a picture of the item. Begin by thoroughly analyzing the image and creating a detailed factual description of what you see. Describe the item with high accuracy, including:

- Color (primary and secondary)
- Material
- Size or relative scale (e.g., handheld, medium, large)
- Shape or form factor
- Distinguishing features such as logos, text, patterns, dents, scratches, tags, stickers, or other markings
- Visible contents (if it is a bag or container)
- Any other visually identifiable characteristics

If the user provides accompanying text, incorporate it only if it is factual and consistent with the image.

After generating this detailed description, immediately output ONLY the structured record below:
(no questions, no explanations, no reasoning)

Subway Location: <station name or null>
Color: <color or colors or null>
Item Category: <category or null>
Item Type: <type or null>
Description: <concise factual summary>

Do not include anything outside the structured record.
Do not ask any follow-up questions.
Do not describe your process.
"""

STANDARDIZER_PROMPT = """
You are the Lost and Found Data Standardizer for a public transit system.
You receive structured text from another model describing an item.
Your task is to map free text fields to standardized tag values and produce a clean JSON record.

Tag Source:
All valid standardized values are in the provided Tags Excel reference summary.
Use only those lists to choose values.

Field rules:

Subway Location:
Compare only with the Subway Location tag list.
Color:
Compare only with the Color tag list.
Item Category:
Compare only with the Item Category tag list.
Item Type:
Compare only with the Item Type tag list.

Use exact or closest textual matches from the correct list only.
If no good match exists return "null" for that field.

Input format:

Subway Location: <value or null>
Color: <value or null>
Item Category: <value or null>
Item Type: <value or null>
Description: <free text description>

Output:

Return only a JSON object of this form:

{
  "subway_location": ["<line or station>", "<line or station>"],
  "color": ["<color1>", "<color2>"],
  "item_category": "<standardized category or null>",
  "item_type": ["<type1>", "<type2>"],
  "description": "<clean description>",
  "time": "<ISO 8601 UTC timestamp>"
}
"""

# ---------------------
# CLIENT SETUP
# ---------------------

def gemini_available() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY"))

# Single shared client
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

def create_operator_chat():
    return client.chats.create(
        model="gemini-2.5-flash",
        history=[
            types.Content(
                role="system",
                parts=[types.Part.from_text(GENERATOR_SYSTEM_PROMPT)]
            )
        ]
    )

def create_user_chat():
    return client.chats.create(
        model="gemini-2.5-flash",
        history=[
            types.Content(
                role="system",
                parts=[types.Part.from_text(USER_SIDE_GENERATOR_PROMPT)]
            )
        ]
    )

# ---------------------
# UTILITIES
# ---------------------

def is_structured_record(text: str) -> bool:
    """Check if the output is a JSON-like structured record"""
    return text.strip().startswith("{") and text.strip().endswith("}")

def extract_field(text: str, field: str) -> str:
    """Extract a single field from structured text"""
    pattern = rf"{field}\s*:\s*(.*)"
    m = re.search(pattern, text)
    return m.group(1).strip() if m else ""

def standardize_description(text: str, tags: dict):
    """Attempt to parse standardized JSON from text"""
    try:
        data = json.loads(text)
    except Exception:
        data = {
            "subway_location": [],
            "color": [],
            "item_category": "null",
            "item_type": [],
            "description": text,
            "time": datetime.now(timezone.utc).isoformat()
        }
    # Ensure list fields are always lists
    for key in ["subway_location","color","item_type"]:
        if key in data and isinstance(data[key], str):
            data[key] = [data[key]]
        elif key not in data:
            data[key] = []
    # Ensure mandatory fields exist
    if "item_category" not in data:
        data["item_category"] = "null"
    if "description" not in data:
        data["description"] = ""
    if "time" not in data:
        data["time"] = datetime.now(timezone.utc).isoformat()
    return data
