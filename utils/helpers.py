import pandas as pd
import streamlit as st
import re

@st.cache_data
def load_tag_data():
    try:
        df = pd.read_excel("Tags.xlsx")
        return {
            "df": df,
            "locations": sorted(set(df["Subway Location"].dropna().astype(str))),
            "colors": sorted(set(df["Color"].dropna().astype(str))),
            "categories": sorted(set(df["Item Category"].dropna().astype(str))),
            "item_types": sorted(set(df["Item Type"].dropna().astype(str))),
        }
    except Exception as e:
        st.error(f"Error loading Tags.xlsx: {e}")
        return None
    
def validate_phone(phone: str) -> bool:
    return bool(re.fullmatch(r"\d{10}", phone or ""))


def validate_email(email: str) -> bool:
    return "@" in (email or "") and "." in (email or "").split("@")[-1]