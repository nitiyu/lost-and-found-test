from pathlib import Path
import os
import streamlit as st
from PIL import Image

from db.postgres import init_db_postgres, get_pg_conn
from db.insert import add_found_item_postgres
from db.search import search_found_items_postgres
from utils.embedding import get_openai_embedding
from utils.gemini import (
    gemini_available,
    create_operator_chat,
    create_user_chat,
    is_structured_record,
    standardize_description,
    extract_field,
)
from utils.helpers import load_tag_data, validate_phone, validate_email

# Ensure DB initialized
if st.secrets.get("PG_CONNECTION_STRING"):
    try:
        init_db_postgres()
    except Exception as e:
        st.error(f"DB init error: {e}")

st.set_page_config(page_title="Lost & Found (Postgres)", layout="wide")
st.title("Lost & Found Intake — Postgres + pgvector + OpenAI")

page = st.sidebar.radio("Go to", ["Upload Found Item (Operator)", "Report Lost Item (User)"])

tag_data = load_tag_data()
if not tag_data:
    st.stop()

# ------------------ Operator ------------------
if page == "Upload Found Item (Operator)":
    st.header("Operator: Upload Found Item (backend-only JSON + DB insert)")

    if not gemini_available():
        st.info("Gemini not configured — automated description disabled.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_image = st.file_uploader("Image of found item", type=["jpg","jpeg","png"]) 
    #with col2:
        #initial_text = st.text_input("Short description (optional)", placeholder="e.g., black backpack with NASA patch")

    if st.button("Start Intake"):
        if not uploaded_image: #and not initial_text
            st.error("Please upload an image.")
        else:
            message_content = ""
            if uploaded_image:
                img = Image.open(uploaded_image).convert("RGB")
                st.image(img, width=200)
                message_content += "I have a photo of the found item. Here is my description based on what I see: "
            #if initial_text:
                #message_content += initial_text

            # Create operator chat
            try:
                operator_chat = create_operator_chat()
                response = operator_chat.send_message(message_content)
                model_text = response.text
            except Exception as e:
                st.error(f"Error calling Gemini: {e}")
                model_text = ""

            if is_structured_record(model_text):
                structured_text = model_text
                final_json = standardize_description(structured_text, tag_data)
                if not final_json:
                    st.error("Failed to standardize the model output.")
                else:
                    # Save image to disk
                    image_path = ""
                    if uploaded_image:
                        images_dir = Path("found_images")
                        images_dir.mkdir(exist_ok=True)
                        image_path = str(images_dir / uploaded_image.name)
                        with open(image_path, "wb") as f:
                            f.write(uploaded_image.getbuffer())

                    contact = st.text_input("Operator contact (optional)")
                    if st.button("Save Found Item to DB"):
                        ok = add_found_item_postgres(final_json, operator_contact=contact or "", image_path=image_path)
                        if ok:
                            st.success("Found item saved to Postgres (JSON handled in backend).")
                        else:
                            st.error("Failed to save found item.")
            else:
                st.info("Model did not emit a final structured record. Model output:")
                st.code(model_text)

# ------------------ User ------------------
if page == "Report Lost Item (User)":
    st.header("User: Report Lost Item")

    if not gemini_available():
        st.info("Gemini not configured — automated structuring disabled.")

    with st.expander("Optional quick info"):
        col1, col2, col3 = st.columns(3)
        with col1:
            location_choice = st.selectbox("Subway station (optional)", [""] + tag_data["locations"])
        with col2:
            category_choice = st.selectbox("Item category (optional)", [""] + tag_data["categories"])
        with col3:
            type_choice = st.selectbox("Item type (optional)", [""] + tag_data["item_types"])

    col_img, col_text = st.columns(2)
    with col_img:
        uploaded_image = st.file_uploader("Image of lost item (optional)", type=["jpg","jpeg","png"], key="user_image")
    with col_text:
        initial_text = st.text_input("Short description", placeholder="e.g., blue iPhone with cracked screen", key="user_text")

    if st.button("Start Report"):
        if not uploaded_image and not initial_text:
            st.error("Please upload an image or enter a short description.")
        else:
            message_text = ""
            if uploaded_image:
                image = Image.open(uploaded_image).convert("RGB")
                st.image(image, width=250)
                message_text += "I have uploaded an image of my lost item. "
            if initial_text:
                message_text += initial_text

            try:
                user_chat = create_user_chat()
                response = user_chat.send_message(message_text)
                model_text = response.text
            except Exception as e:
                st.error(f"Error calling Gemini: {e}")
                model_text = ""

            if is_structured_record(model_text):
                structured_text = model_text
                merged_text = f"""
Subway Location: {location_choice or extract_field(structured_text, 'Subway Location')}
Color: {extract_field(structured_text, 'Color')}
Item Category: {category_choice or extract_field(structured_text, 'Item Category')}
Item Type: {type_choice or extract_field(structured_text, 'Item Type')}
Description: {extract_field(structured_text, 'Description')}
"""
                st.markdown("### Final merged record (used for search)")
                st.code(merged_text)

                final_json = standardize_description(merged_text, tag_data)
                if not final_json:
                    st.error("Failed to standardize.")
                else:
                    st.success("Standardized record generated (not saved as DB entry).")
                    contact = st.text_input("Phone number, ten digits")
                    email = st.text_input("Email address")
                    if st.button("Submit Lost Item Report and Search for Matches"):
                        if not validate_phone(contact):
                            st.error("Please enter a ten digit phone number without spaces.")
                        elif not validate_email(email):
                            st.error("Please enter a valid email address.")
                        else:
                            matches = search_found_items_postgres(final_json, k=5)
                            if not matches:
                                st.info("No matches found.")
                            else:
                                st.markdown(f"### Top {len(matches)} matches (tag-filtered + vector-ranked)")
                                for m in matches:
                                    st.write(f"Similarity: {m['similarity']:.4f}  —  Category: {m['item_category']}  —  Location: {', '.join(m['subway_location'])}")
                                    st.write(m["description"])
                                    if m["image_path"]:
                                        try:
                                            st.image(m["image_path"], width=200)
                                        except Exception:
                                            st.text("Image path present but could not be displayed.")
                                    st.markdown("---")
            else:
                st.info("Model did not emit a final structured record. Model output:")
                st.code(model_text)



